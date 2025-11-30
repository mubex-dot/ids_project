import json
import time
import threading
import pandas as pd
from queue import Queue
from app.models.infer import predict
from app.features.columns_nsl_kdd import EXPECTED_FEATURES  

def extract_features(alert: dict) -> dict:
    """Extract only the needed features for the model."""
    return {
        "protocol_type": alert.get("proto", "unknown"),
        "service": alert.get("app_proto", "unknown"),
        "flag": alert.get("tcp_flags", "OTH"),
        "src_bytes": alert.get("tx_bytes", 0),
        "dst_bytes": alert.get("rx_bytes", 0),
    }

def normalize_sample(sample: dict) -> dict:
    norm = {}
    for f in EXPECTED_FEATURES:
        norm[f] = sample.get(f, "unknown")
    return norm

def worker(log_path: str, model_path: str, queue: Queue):
    with open(log_path, 'r', encoding='utf-8') as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            try:
                alert = json.loads(line.strip())
                sample = extract_features(alert)
                norm = normalize_sample(sample)
                res = predict(model_path, norm)
                queue.put({"sample": norm, "prediction": res})
            except Exception:
                continue

def start_suricata_monitor(log_path: str, model_path: str):
    q = Queue()
    threading.Thread(target=worker, args=(log_path, model_path, q), daemon=True).start()
    return q
