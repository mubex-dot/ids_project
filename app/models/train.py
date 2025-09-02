import argparse
from pathlib import Path
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from app.features.columns_nsl_kdd import *
from app.utils.io import *

INTERIM = Path("data/interim")
MODELS = Path("models")

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_col = [c for c in CATEGORICAL if c in X.columns]
    numerical_col = [c for c in X.columns if c not in categorical_col + ["label", "target"]] # All columns except categorical + label and added target columns
    pre = ColumnTransformer([
        ("categorical_col", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_col),
        ("numerical_col", StandardScaler(), numerical_col),
    ])
    return pre

def grid_search(model_name: str, pre: ColumnTransformer, X, y):
    if model_name == "svm":
        clf = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
        grid = {"clf__C": [1, 3], "clf__gamma": ["scale"]}
    elif model_name == "dt":
        clf = DecisionTreeClassifier(
            criterion="gini",
            class_weight="balanced",
            random_state=42,
        )
        grid = {
            "clf__max_depth": [None, 20, 40],
            "clf__min_samples_split": [2, 10, 50],
            "clf__min_samples_leaf": [1, 5, 10],
        }
    else:
        raise ValueError("Unknown model name")

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X, y)
    return gs


def main(task="binary", models=["svm", "dt"], fast=False):
    ensure_dir(MODELS)
    df = pd.read_csv(INTERIM / f"train_{task}.csv")
    if fast and len(df) > 40000:
        df = df.sample(40000, random_state=42)
    X = df.drop(columns=["target"])
    y = df["target"].values
    pre = build_preprocessor(X)
    for name in models:
        print(f"\n=== Training {name.upper()} ===")
        gs = grid_search(name, pre, X, y)
        print("Best params:", gs.best_params_)
        best = gs.best_estimator_
        out = MODELS / (f"best_{name}.joblib")
        dump(best, out)
        print("Saved:", out)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
#     parser.add_argument("--models", nargs="+", choices=["svm", "dt"], default=["svm", "dt"])
#     parser.add_argument("--fast", action="store_true", help="Use a subset of training rows (faster SVM)")
#     args = parser.parse_args()
#     main(task=args.task, models=args.models, fast=args.fast)
