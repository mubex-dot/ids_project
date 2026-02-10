#!/usr/bin/env bash
set -euo pipefail

python src/data/download_nsl_kdd.py
python src/data/make_dataset.py --task binary
python src/app/models/train.py --task binary --models svm dt --fast
python src/app/models/evaluate.py --task binary --model both
