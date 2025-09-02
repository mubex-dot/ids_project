"""
Main entry point for the IDS pipeline.
Runs: download, preprocess, feature extraction, train, evaluate, and inference.
"""
from app.data import download_nsl_kdd, make_dataset
from app.models import train, evaluate, infer
# import sys


def main():
    print("[1/5] Downloading NSL-KDD dataset...")
    download_nsl_kdd.main() # NOTE: If keeps failing, download manually and export to data/raw folder

    print("[2/5] Preprocessing dataset...")
    # Always use binary task for pipeline; change to 'multiclass' if needed
    make_dataset.main(task="binary")

    print("[3/5] Training models (SVM and Decision Tree)...")
    train.main(task="binary", models=["svm", "dt"], fast=True)

    print("[4/5] Evaluating models...")
    evaluate.main(task="binary", model="both")

    print("[5/5] IDS pipeline complete. Models and reports are saved.")

if __name__ == "__main__":
    main()
