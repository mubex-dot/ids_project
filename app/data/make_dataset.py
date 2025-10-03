import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from app.features.columns_nsl_kdd import *
from app.utils.io import *

RAW = Path("data/raw")
INTERIM = Path("data/interim")

ATTACK_TOKEN = "normal"

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if DIFFICULTY_COL in df.columns:
        df = df.drop(columns=[DIFFICULTY_COL])
        
    df[LABEL_COL] = df[LABEL_COL].str.lower()
    return df

def add_targets(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "binary":
        df["target"] = np.where(df[LABEL_COL] == ATTACK_TOKEN, 0, 1)
    elif task == "multiclass":
        families = df[LABEL_COL].apply(lambda x: x if x == ATTACK_TOKEN else FAMILY_MAP.get(x, "OtherAttack"))
        # Map to integers consistently
        cat_order = sorted(families.unique())
        to_int = {c: i for i, c in enumerate(cat_order)}
        df["target"] = families.map(to_int)
        # Save a tiny label map for later use
        (INTERIM / "label_map_multiclass.json").write_text(pd.Series({int(v): k for k, v in to_int.items()}).to_json())
    else:
        raise ValueError("task must be 'binary' or 'multiclass'")
    return df


def main(task):
    print(f"[make_dataset] Would process dataset for task: {task}")
    ensure_dir(INTERIM)

    tr = pd.read_csv(RAW / "KDDTrain+.csv")
    te = pd.read_csv(RAW / "KDDTest+.csv")

    tr = add_targets(clean(tr), task)
    te = add_targets(clean(te), task)

    print("Saving cleaned datasets to interim folder")
    tr.to_csv(INTERIM / f"train_{task}.csv", index=False)
    te.to_csv(INTERIM / f"test_{task}.csv", index=False)
    print("Wrote:", INTERIM / f"train_{task}.csv"," and ", INTERIM / f"test_{task}.csv")
