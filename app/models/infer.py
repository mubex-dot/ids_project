import argparse
import json
import sys
from joblib import load
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .joblib model pipeline")
    parser.add_argument("--json", help="Single JSON sample (string)")
    parser.add_argument("--stdin", action="store_true", help="Read newline-delimited JSON from stdin")
    args = parser.parse_args()

    clf = load(args.model)

    def predict_one(d):
        # Ensure required categorical keys exist; missing numerics default to 0
        for k in ["protocol_type","service","flag"]:
            d.setdefault(k, "unknown")
        df = pd.DataFrame([d])
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

    if args.stdin:
        for line in sys.stdin:
            if not line.strip():
                continue
            d = json.loads(line)
            print(json.dumps(predict_one(d)))
    elif args.json:
        d = json.loads(args.json)
        print(json.dumps(predict_one(d)))
    else:
        parser.error("Provide --json or --stdin")

if __name__ == "__main__":
    main()
