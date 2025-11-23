from app.data import download_nsl_kdd, make_dataset
from app.models import train, evaluate


def main():
    print("[1/5] Downloading NSL-KDD dataset...")
    # NOTE: If keeps failing, download manually and add to data/raw folder
    # download_nsl_kdd.main() 

    print("[2/5] Preprocessing dataset...")
    # NOTE: Tasks can either be binary or binary
    make_dataset.main(task="binary")

    print("[3/5] Training models (SVM and Decision Tree)...")
    train.main(task="binary", models=["svm", "dt"], fast=True)

    print("[4/5] Evaluating models...")
    # Model can either be "svm", "dt" or both
    evaluate.main(task="binary", model="both")

    print("[5/5] IDS pipeline complete. Models and reports are saved.")

if __name__ == "__main__":
    main()
