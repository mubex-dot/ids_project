from __future__ import annotations
from pathlib import Path
import time
import requests
import pandas as pd

# keep your existing imports for COLUMNS (and ensure_dir if you already have it)
from app.features.columns_nsl_kdd import COLUMNS
try:
    from app.utils.io import ensure_dir
except Exception:
    # fallback if ensure_dir isn't available
    def ensure_dir(p: Path):
        p.mkdir(parents=True, exist_ok=True)

RAW = Path("data/raw")
TRAIN_FILE = RAW / "KDDTrain+.txt"
TEST_FILE  = RAW / "KDDTest+.txt"

# Primary + mirror URLs (order matters; fastest first)
URLS_TRAIN = [
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "https://raw.githubusercontent.com/MathCoub/NSL-KDD/master/KDDTrain+.txt",
]
URLS_TEST = [
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
    "https://raw.githubusercontent.com/MathCoub/NSL-KDD/master/KDDTest+.txt",
]

def download_with_retries(urls, out_path: Path, max_retries: int = 4, timeout: int = 60) -> None:
    """Try each url with retries/backoff; write to out_path. Raises on failure."""
    headers = {"User-Agent": "NSL-KDD-downloader/1.0"}
    for url in urls:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[download] {url} (attempt {attempt}/{max_retries})")
                with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
                    r.raise_for_status()
                    tmp = out_path.with_suffix(out_path.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                    tmp.replace(out_path)
                return
            except Exception as e:
                wait = min(2 ** attempt, 30)
                print(f"  -> failed: {e} ; retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Could not download {out_path.name} from any mirror.")

def read_nsl_kdd_txt(path: Path) -> pd.DataFrame:
    # Be forgiving across forks; some lines can be quirky.
    return pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        engine="python",
        on_bad_lines="skip",
    )

def main():
    ensure_dir(RAW)

    # Download if not cached
    if not TRAIN_FILE.exists():
        print("[1/4] Downloading training file …")
        download_with_retries(URLS_TRAIN, TRAIN_FILE)
    else:
        print("[1/4] Using cached:", TRAIN_FILE)

    if not TEST_FILE.exists():
        print("[2/4] Downloading test file …")
        download_with_retries(URLS_TEST, TEST_FILE)
    else:
        print("[2/4] Using cached:", TEST_FILE)

    # Read locally, then save CSVs
    print("[3/4] Parsing to DataFrames …")
    df_tr = read_nsl_kdd_txt(TRAIN_FILE)
    df_te = read_nsl_kdd_txt(TEST_FILE)

    if df_tr.empty or df_te.empty:
        raise RuntimeError("Downloaded files are empty or malformed.")

    print("[4/4] Saving CSVs …")
    out_train_csv = RAW / "KDDTrain+.csv"
    out_test_csv  = RAW / "KDDTest+.csv"
    df_tr.to_csv(out_train_csv, index=False)
    df_te.to_csv(out_test_csv, index=False)

    print("Done. Saved:", out_train_csv, out_test_csv)
    print("Train shape:", df_tr.shape, "| Test shape:", df_te.shape)

if __name__ == "__main__":
    main()
