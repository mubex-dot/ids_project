from joblib import load
import pandas as pd
import os
import logging
from typing import List, Dict, Any, Union

try:
    # Local feature definitions
    from app.features.columns_nsl_kdd import COLUMNS as NSL_COLUMNS, CATEGORICAL as NSL_CATEGORICAL
except ImportError:
    # Fallback if package import not available
    NSL_COLUMNS = []
    NSL_CATEGORICAL = ['protocol_type', 'service', 'flag']

# Configure logging
logger = logging.getLogger(__name__)


def predict(model: Union[str, object], sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict on a single sample dictionary.
    
    Args:
        model: Path to model file or loaded model object
        sample_dict: Dictionary with feature values
        
    Returns:
        Dictionary with prediction and score:
        {
            "prediction": 0 (normal) or 1 (attack),
            "score_attack": float (probability of attack if available)
        }
    """
    # Load model if path is provided
    if isinstance(model, (str, bytes, os.PathLike)):
        try:
            clf = load(model)
        except Exception as e:
            logger.error("Failed to load model from %s: %s", model, e)
            raise
    else:
        # Assume it's already a loaded estimator
        clf = model

    # Ensure required categorical features exist
    for k in NSL_CATEGORICAL:
        sample_dict.setdefault(k, "other")

    # Try to infer expected input columns from the pipeline
    expected_cols: List[str] = []
    try:
        if hasattr(clf, 'named_steps'):
            for name, step in clf.named_steps.items():
                # ColumnTransformer stores transformers_ with (name, transformer, columns)
                if hasattr(step, 'transformers_'):
                    for _, _, cols in step.transformers_:
                        if isinstance(cols, (list, tuple)):
                            expected_cols.extend(cols)
                    break
    except Exception as e:
        logger.debug("Could not infer columns from pipeline: %s", e)

    # Fallback to known NSL columns if pipeline introspection failed
    if not expected_cols:
        expected_cols = [col for col in NSL_COLUMNS if col not in ['label', 'difficulty']]

    # Build a sanitized row containing all expected columns with safe defaults
    row = {}
    for col in expected_cols:
        if col in sample_dict:
            row[col] = sample_dict[col]
        else:
            # Categorical defaults
            if col in NSL_CATEGORICAL:
                row[col] = 'other'
            else:
                # Numeric default
                row[col] = 0

    # Create DataFrame
    df = pd.DataFrame([row])
    
    # Ensure numeric columns have numeric dtype and NaNs filled
    for col in df.columns:
        if df[col].dtype.kind in "biufc":  # Bool, int, uint, float, complex
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Optional debug logging
    if os.environ.get("IDS_DEBUG") in ("1", "true", "yes"):
        logger.info("[infer] Model type: %s", type(clf))
        
        try:
            if hasattr(clf, "named_steps"):
                logger.info("[infer] Pipeline steps: %s", list(clf.named_steps.keys()))
        except Exception:
            pass
        
        try:
            if hasattr(clf, "feature_names_in_"):
                logger.info("[infer] Expected features: %s", clf.feature_names_in_)
        except Exception:
            pass
        
        logger.info("[infer] Input features: %s", list(row.keys()))
        logger.info("[infer] Missing features: %s", 
                   [col for col in expected_cols if col not in sample_dict])
        logger.info("[infer] DataFrame shape: %s", df.shape)
        logger.info("[infer] Sample values: %s", row)

    # Make prediction
    try:
        prediction = int(clf.predict(df)[0])
    except ValueError as e:
        # If columns are still missing, attempt to add defaults and retry
        missing_msg = str(e)
        logger.error("Prediction failed due to columns: %s", missing_msg)
        
        # Add any missing NSL columns and retry
        missing_cols = [col for col in NSL_COLUMNS 
                       if col not in df.columns and col not in ['label', 'difficulty']]
        if missing_cols:
            for col in missing_cols:
                df[col] = 0
            try:
                prediction = int(clf.predict(df)[0])
            except Exception as retry_error:
                logger.exception("Prediction retry failed")
                raise retry_error
        else:
            raise

    # Prepare output
    result = {"prediction": prediction}

    # Get probability if available
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(df)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                # Class 0 = normal, Class 1 = attack
                # Note: In binary classification, we expect 2 classes
                if proba.shape[1] == 2:
                    # Binary classification
                    result["score_attack"] = float(proba[0, 1])
                else:
                    # Multiclass - probability of positive class
                    result["score_attack"] = float(proba[0, -1])
            elif proba.ndim == 2 and proba.shape[1] == 1:
                # Some models might output single class probability
                result["score_attack"] = float(proba[0, 0])
        except Exception as e:
            logger.warning("predict_proba failed: %s", e)
    elif hasattr(clf, "decision_function"):
        try:
            decision = clf.decision_function(df)
            # Normalize decision scores to [0, 1] for consistency
            if decision.ndim == 1:
                # Binary classification
                from scipy.special import expit
                result["score_attack"] = float(expit(decision[0]))
            else:
                # Multiclass - use max probability
                from scipy.special import softmax
                probabilities = softmax(decision[0])
                result["score_attack"] = float(probabilities[-1])
        except Exception as e:
            logger.warning("decision_function failed: %s", e)

    # Debug log the result
    if os.environ.get("IDS_DEBUG") in ("1", "true", "yes"):
        logger.info("[infer] Prediction: %s", result)

    return result


def batch_predict(model: Union[str, object], samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict on multiple samples.
    
    Args:
        model: Path to model file or loaded model object
        samples: List of dictionaries with feature values
        
    Returns:
        List of prediction results
    """
    results = []
    for sample in samples:
        try:
            result = predict(model, sample)
            results.append(result)
        except Exception as e:
            logger.error("Failed to predict sample %s: %s", sample, e)
            results.append({"error": str(e)})
    return results


def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a sample dictionary.
    
    Args:
        sample: Input sample dictionary
        
    Returns:
        Normalized sample with all required features
    """
    normalized = sample.copy()
    
    # Ensure categorical features exist
    for cat in NSL_CATEGORICAL:
        if cat not in normalized:
            normalized[cat] = "other"
    
    # Ensure required numeric features exist
    required_numeric = [
        'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'is_guest_login', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate'
    ]
    
    for num in required_numeric:
        if num not in normalized:
            normalized[num] = 0
        else:
            # Convert to numeric
            try:
                normalized[num] = float(normalized[num])
            except (ValueError, TypeError):
                normalized[num] = 0
    
    return normalized


if __name__ == "__main__":
    import sys
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from app.features.columns_nsl_kdd import COLUMNS, CATEGORICAL
    
    # Test with a model path (using dt model first)
    test_model_path = "models/best_dt.joblib" 
    
    if os.path.exists(test_model_path):
        print("Testing inference with model:", test_model_path)
        
        # Create a test sample
        test_sample = {
            "duration": 0.1,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 100,
            "dst_bytes": 2000,
            "count": 1,
            "srv_count": 1
        }
        
        # Fill missing features
        test_sample = validate_sample(test_sample)
        
        print("Test sample features:", len(test_sample))
        print("Required features:", len([c for c in COLUMNS if c not in ['label', 'difficulty']]))
        
        # Make prediction
        result = predict(test_model_path, test_sample)
        print("Prediction result:", result)
        
        # Test with attack-like traffic
        test_sample_attack = test_sample.copy()
        test_sample_attack.update({
            "src_bytes": 1000000,
            "dst_bytes": 50,
            "count": 100,
            "srv_count": 100,
            "hot": 10,
            "num_failed_logins": 5
        })
        
        result_attack = predict(test_model_path, test_sample_attack)
        print("Attack prediction result:", result_attack)
    else:
        print(f"Model not found at {test_model_path}")
        print("Run main.py to train a model first.")