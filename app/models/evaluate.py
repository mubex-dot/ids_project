import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

from app.utils.io import *

INTERIM = Path("data/interim")
MODELS = Path("models")
REPORTS = Path("reports/metrics")

def plot_confusion(cm, labels, outpath):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main(task="binary", model="both"):
    ensure_dir(REPORTS)
    df_te = pd.read_csv(INTERIM / f"test_{task}.csv")
    X_te = df_te.drop(columns=["target"])
    y_te = df_te["target"].to_numpy()
    model_names = ([model] if model != "both" else ["svm","dt"])
    for name in model_names:
        clf = load(Path("models") / f"best_{name}.joblib")
        y_pred = clf.predict(X_te)
        y_pred = np.asarray(y_pred)
        metrics = {}
        metrics["accuracy"] = float(accuracy_score(y_te, y_pred))
        pr, rc, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
        metrics.update({"precision_macro": float(pr), "recall_macro": float(rc), "f1_macro": float(f1)})
        try:
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_te)
                if proba.ndim == 1 or proba.shape[1] == 2:
                    pos = proba[:, 1] if proba.ndim > 1 else proba
                    metrics["roc_auc_ovr"] = float(roc_auc_score(np.asarray(y_te), np.asarray(pos)))
                else:
                    metrics["roc_auc_ovr"] = float(roc_auc_score(np.asarray(y_te), np.asarray(proba), multi_class="ovr"))
            else:
                scores = clf.decision_function(X_te)
                if scores.ndim == 1:
                    metrics["roc_auc_ovr"] = float(roc_auc_score(np.asarray(y_te), np.asarray(scores)))
                else:
                    metrics["roc_auc_ovr"] = float(roc_auc_score(np.asarray(y_te), np.asarray(scores), multi_class="ovr"))
        except Exception:
            pass
        cm = confusion_matrix(np.asarray(y_te), np.asarray(y_pred))
        labels = ["0","1"] if task=="binary" else [str(i) for i in sorted(np.unique(np.asarray(y_te)))]
        plot_confusion(cm, labels, REPORTS / f"cm_{name}_{task}.png")
        report = classification_report(np.asarray(y_te), np.asarray(y_pred), zero_division=0)
        with open(REPORTS / f"report_{name}_{task}.txt", "w", encoding="utf-8") as f:
            f.write(str(report))
        save_json(metrics, REPORTS / f"metrics_{name}_{task}.json")
        print(f"Saved metrics for {name} →", REPORTS / f"metrics_{name}_{task}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--model", choices=["svm", "dt", "both"], default="both")
    args = parser.parse_args()
    main(task=args.task, model=args.model)
