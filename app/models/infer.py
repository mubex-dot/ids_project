from joblib import load
import pandas as pd
import os
import logging


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
    for k in ["protocol_type", "service", "flag"]:
        sample_dict.setdefault(k, "unknown")

    df = pd.DataFrame([sample_dict])
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

    y = int(clf.predict(df)[0])
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
