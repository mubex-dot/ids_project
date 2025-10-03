from joblib import load
import pandas as pd

def predict(model_path, sample_dict):
    
    clf = load(model_path)
    # Ensure required categorical keys exist; missing numerics default to 0
    for k in ["protocol_type","service","flag"]:
        sample_dict.setdefault(k, "unknown")
    df = pd.DataFrame([sample_dict])
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(0)
    y = int(clf.predict(df)[0])
    out = {"prediction": y}
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(df)
        if proba.ndim == 2 and proba.shape[1] > 1:
            out["score_attack"] = int(proba[0, -1])
    return out
