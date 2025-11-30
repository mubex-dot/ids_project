

import argparse
import json
import time
import threading
from collections import deque, defaultdict
from queue import Queue
from datetime import datetime, timezone
import os
import joblib
import pandas as pd

# Config
DEFAULT_MODEL = "models/best_dt.joblib"
DEFAULT_EVE = "/var/log/suricata/eve.json"
DEFAULT_ALERT_FILE = "ids_alerts.jsonl"
NUM_WORKERS = 4
SLIDING_WINDOW = 2.0

# Sliding windows (module-level so workers share)
_by_host = defaultdict(deque)
_by_host_service = defaultdict(deque)
_alert_queue = Queue()
_recent_alerts = deque(maxlen=500)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--eve", default=DEFAULT_EVE)
    ap.add_argument("--alert-file", default=DEFAULT_ALERT_FILE)
    ap.add_argument("--threads", type=int, default=NUM_WORKERS)
    ap.add_argument("--window", type=float, default=SLIDING_WINDOW)
    return ap.parse_args()

def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(k, str) and k in d:
            return d[k]
    return default

def extract_features(rec, window):
    ts_str = safe_get(rec, "timestamp", "ts")
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() if ts_str else time.time()
    except Exception:
        ts = time.time()

    src_ip = safe_get(rec, "src_ip", "src")
    dst_ip = safe_get(rec, "dest_ip", "dst") or "unknown"
    src_port = safe_get(rec, "src_port", "sp")
    dst_port = safe_get(rec, "dest_port", "dp")

    protocol = safe_get(rec, "proto", "protocol") or (rec.get("flow") or {}).get("proto") or "other"
    service = safe_get(rec, "app_proto", "service") or (rec.get("flow") or {}).get("service") or "other"
    flag = safe_get(rec, "tcp_flags", "flags") or (rec.get("flow") or {}).get("tcp_flags") or "OTH"

    src_bytes = safe_get(rec, "tx_bytes", "src_bytes") or (rec.get("flow") or {}).get("bytes_toserver", 0)
    dst_bytes = safe_get(rec, "rx_bytes", "dst_bytes") or (rec.get("flow") or {}).get("bytes_toclient", 0)
    try: src_bytes = int(src_bytes)
    except: src_bytes = 0
    try: dst_bytes = int(dst_bytes)
    except: dst_bytes = 0

    duration = safe_get(rec, "duration") or (rec.get("flow") or {}).get("age", 0.0)
    try: duration = float(duration)
    except: duration = 0.0

    # sliding window counts
    dq = _by_host[dst_ip]
    dq.append(ts)
    while dq and ts - dq[0] > window:
        dq.popleft()
    count = len(dq)

    dq2 = _by_host_service[(dst_ip, service)]
    dq2.append(ts)
    while dq2 and ts - dq2[0] > window:
        dq2.popleft()
    srv_count = len(dq2)

    sample = {
        "protocol_type": str(protocol).lower(),
        "service": str(service),
        "flag": str(flag),
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "count": count,
        "srv_count": srv_count,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "timestamp": ts_str
    }
    return sample

def worker(pipeline, expected_cols, window, alert_file):
    while True:
        rec = _alert_queue.get()
        try:
            sample = extract_features(rec, window)
            # Ensure DataFrame contains all expected columns the pipeline needs
            row = {}
            cols = expected_cols or [
                "protocol_type", "service", "flag", "duration",
                "src_bytes", "dst_bytes", "count", "srv_count"
            ]
            for c in cols:
                if c in sample and sample[c] is not None:
                    row[c] = sample[c]
                else:
                    # preserve categorical defaults
                    if c in ("protocol_type", "service", "flag"):
                        row[c] = "other"
                    else:
                        row[c] = 0
            df = pd.DataFrame([row])
            # coerce numeric columns
            for c in df.columns:
                try:
                    if isinstance(row.get(c), (int, float)):
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                except Exception:
                    pass

            try:
                pred = pipeline.predict(df)[0]
            except ValueError as e:
                # Attempt a best-effort fix: add any missing NSL columns and retry once
                msg = str(e)
                missing = []
                if "columns are missing" in msg:
                    try:
                        missing = eval(msg.split(':',1)[1].strip())
                    except Exception:
                        missing = []
                for m in missing:
                    df[m] = 0
                pred = pipeline.predict(df)[0]
            if int(pred) == 1:
                alert = {**sample, "pred": "ATTACK", "ts": sample.get("timestamp") or datetime.now(timezone.utc).isoformat()}
                print("[ALERT]", alert)
                _recent_alerts.appendleft(alert)
                with open(alert_file, "a", buffering=1) as fh:
                    fh.write(json.dumps(alert) + "\n")
        except Exception:
            import traceback; traceback.print_exc()
        finally:
            _alert_queue.task_done()

def tail_f(path):
    while True:
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(0, 2)
                try: inode = os.fstat(f.fileno()).st_ino
                except: inode = None
                while True:
                    line = f.readline()
                    if line:
                        yield line
                    else:
                        time.sleep(0.1)
                        try:
                            if inode and os.stat(path).st_ino != inode:
                                break
                        except FileNotFoundError:
                            break
        except FileNotFoundError:
            time.sleep(0.5)

def monitor(eve_path):
    for line in tail_f(eve_path):
        try:
            rec = json.loads(line)
        except:
            continue
        _alert_queue.put(rec)

def main():
    args = parse_args()
    pipeline = joblib.load(args.model)
    # best-effort extract expected_cols from pipeline (if ColumnTransformer present)
    expected_cols = None
    try:
        for name, step in pipeline.named_steps.items():
            if hasattr(step, "transformers_"):
                cols = []
                for _, _, c in step.transformers_:
                    if isinstance(c, (list, tuple)):
                        cols.extend(c)
                expected_cols = cols
                break
    except Exception:
        expected_cols = ["protocol_type","service","flag","duration","src_bytes","dst_bytes","count","srv_count"]

    # start workers
    for _ in range(args.threads):
        t = threading.Thread(target=worker, args=(pipeline, expected_cols, args.window, args.alert_file), daemon=True)
        t.start()

    print(f"[+] Monitoring {args.eve} (window={args.window}s) - press Ctrl+C to stop")
    monitor(args.eve)

if __name__ == "__main__":
    main()
