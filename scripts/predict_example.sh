#!/usr/bin/env bash
python src/models/infer.py --model models/best_dt.joblib --json '{
  "protocol_type":"tcp", "service":"http", "flag":"SF",
  "src_bytes": 181, "dst_bytes": 5450, "count": 2, "srv_count": 2
}'
