from joblib import load
import pandas as pd
import os
import logging
from typing import List
try:
    # local feature definitions
    from app.features.columns_nsl_kdd import COLUMNS as NSL_COLUMNS, CATEGORICAL as NSL_CATEGORICAL
except Exception:
    # fallback if package import not available (script run directly)
    NSL_COLUMNS = []
    NSL_CATEGORICAL = ['protocol_type', 'service', 'flag']


def predict(model_path, sample_dict):
    """Load model and predict on a single sample dict.

    - Ensures categorical keys exist.
    - Fills numeric NaNs with 0.
    - Returns dict: {prediction: 0|1, score_attack: float (if available)}
    - When `IDS_DEBUG=1` is set in the environment, logs the input DataFrame and model info.
    """
    logger = logging.getLogger(__name__)
    clf = load(model_path)

    # Ensure required categorical keys exist; missing numerics default to 0
    for k in NSL_CATEGORICAL:
        sample_dict.setdefault(k, "unknown")

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
    except Exception:
        expected_cols = []

    # Fallback to known NSL columns if pipeline introspection failed
    if not expected_cols:
        expected_cols = NSL_COLUMNS or ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']

    # Build a sanitized row containing all expected columns with safe defaults
    row = {}
    for c in expected_cols:
        if c in sample_dict:
            row[c] = sample_dict[c]
        else:
            # categorical defaults
            if c in NSL_CATEGORICAL:
                row[c] = 'other'
            else:
                # numeric default
                row[c] = 0

    df = pd.DataFrame([row])
    # Ensure numeric columns have numeric dtype and NaNs filled
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(0)

    # Optional debug: show model and input
    try:
        if os.environ.get("IDS_DEBUG") in ("1", "true", "yes"):
            logger.info("[infer] model=%s", type(clf))
            # Try to show pipeline steps / feature names if available
            try:
                if hasattr(clf, "named_steps"):
                    logger.info("[infer] pipeline steps=%s", list(clf.named_steps.keys()))
            except Exception:
                pass
            try:
                fn = getattr(clf, "feature_names_in_", None)
                logger.info("[infer] feature_names_in_=%s", fn)
            except Exception:
                pass
            logger.info("[infer] input_df=%s", df.to_dict(orient="records"))
    except Exception:
        logger.exception("Failed while logging debug info")

    try:
        y = int(clf.predict(df)[0])
    except ValueError as e:
        # If columns are still missing, attempt to add NSL_COLUMNS defaults and retry once
        missing_msg = str(e)
        logger.error("Prediction failed due to columns: %s", missing_msg)
        if NSL_COLUMNS:
            for c in NSL_COLUMNS:
                if c not in df.columns:
                    df[c] = 0
            try:
                y = int(clf.predict(df)[0])
            except Exception:
                logger.exception("Prediction retry failed")
                raise
        else:
            raise
    out = {"prediction": y}
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(df)
            if proba.ndim == 2 and proba.shape[1] > 1:
                # Return probability for the positive/attack class as float
                out["score_attack"] = float(proba[0, -1])
        except Exception:
            logger.exception("predict_proba failed")
    return out
