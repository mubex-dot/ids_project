# Network-based IDS — SVM & Decision Tree (from scratch)

This is a complete, opinionated starter you can clone into a folder and run end‑to‑end. It:

- Downloads **NSL‑KDD** (open academic IDS dataset) automatically
- Preprocesses (one‑hot for categorical + scaling for numeric)
- Trains **SVM (RBF)** and **Decision Tree** with stratified CV
- Evaluates on the canonical **KDDTest+** split (no leakage)
- Saves ready‑to‑use pipelines (`.joblib`) and exposes a simple CLI for inference

> Later, you can swap in UNSW‑NB15 or CIC‑IDS2017 by changing `DATASET` in `config.yaml` and updating column maps.

---

## 0) Folder layout

```
ids_svm_dt/
├─ README.md                 # this guide
├─ requirements.txt
├─ config.yaml               # hyperparams & paths in one place
├─ main.py                   # single entry point for the full pipeline
├─ app/
│  ├─ data/
│  │  ├─ download_nsl_kdd.py
│  │  └─ make_dataset.py
│  ├─ features/
│  │  └─ columns_nsl_kdd.py
│  ├─ models/
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  └─ infer.py
│  └─ utils/
│     └─ io.py
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ models/
├─ reports/
│  └─ metrics/
└─ scripts/
   ├─ run_all.sh
   └─ predict_example.sh
```

---

## 1) Quickstart

```bash
# 1) Create env & install deps
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) Run the full IDS pipeline (download, preprocess, train, evaluate, infer)
python main.py

# 3) (Optional) Predict on a single JSON sample
python app/models/infer.py --model models/best_svm.joblib --json '{"protocol_type":"tcp","service":"http","flag":"SF"}'
```

---

## 3) Notes, decisions & how to extend

**Why NSL-KDD first?** Small, fast to iterate, and standard for classroom/academic IDS baselines. We keep the official `KDDTest+` as the evaluation set to avoid leakage.

**Binary vs Multiclass**  
Start with `--task binary` (normal vs attack). When stable, try `--task multiclass` to predict attack families (DoS/Probe/R2L/U2R). The code already supports it.

**Imbalance handling**  
We enabled `class_weight="balanced"` in both models. You can later add SMOTE (especially helpful for tree-based ensembles) — we didn’t include SMOTE here to keep dependencies slim and preserve the official split.

**Speed**  
SVM on full NSL‑KDD can be heavy on older laptops. Use `--fast` (subsamples ~40k rows). After validation, retrain without `--fast` for best scores.

**Thresholding**  
For binary, use `score_attack` from SVM `predict_proba` as a risk score. A practical starting alert threshold is 0.5; tune it using PR curves for your tolerance of false positives.

**Going live**

- Export NetFlow/Zeek to CSV/JSON, map to the same feature names (protocol/service/flag + numerics).
- Load `models/best_*.joblib` and call `predict`/`predict_proba` in a small FastAPI app.
- Write alerts into Kafka/Redis or a DB table; visualize in a dashboard (Superset/Grafana).

**Upgrade paths**

- Swap Decision Tree → RandomForest/XGBoost.
- Add **calibration** (`CalibratedClassifierCV`) for better probability estimates.
- Feature selection with mutual information to slim the SVM input space.

**Reproducibility**  
We pin a `seed` in `config.yaml`. All splits and CV folds are stratified.

---

## 4) License & dataset attribution

- Code here is MIT‑style by default; adapt as needed.
- **NSL‑KDD** courtesy of the University of New Brunswick / Canadian Institute for Cybersecurity; this starter downloads public mirrors for convenience.
