import argparse
import json
import time
import threading
from collections import deque, defaultdict
from queue import Queue
from datetime import datetime, timezone
import os
import sys
import random

import joblib
# Ensure project root is on sys.path so `from app...` imports work when running this script directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from app.models.infer import predict
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

    # BETTER DEFAULT VALUES FOR CATEGORICAL FEATURES
    # Model was trained on specific values, not "other"
    
    # Protocol: Map to "tcp", "udp", or "icmp" only
    raw_proto = safe_get(rec, "proto", "protocol") or (rec.get("flow") or {}).get("proto") or ""
    protocol = "tcp"  # Default to most common
    proto_lower = raw_proto.lower()
    if proto_lower in ["tcp", "udp", "icmp"]:
        protocol = proto_lower
    
    # Service: Map to common NSL-KDD services
    raw_service = safe_get(rec, "app_proto", "service") or (rec.get("flow") or {}).get("service") or ""
    service = "http"  # Default to most common
    
    # Map Suricata services to NSL-KDD
    service_map = {
        "http": "http", "https": "http", "dns": "domain_u", "ftp": "ftp_data",
        "ssh": "ssh", "smtp": "smtp", "tls": "ssl", "ssl": "ssl",
        "dhcp": "dhcp", "snmp": "snmp", "telnet": "telnet"
    }
    
    if raw_service in service_map:
        service = service_map[raw_service]
    elif raw_service and "data" in raw_service.lower():
        service = "ftp_data"  # Generic data service
    else:
        # Guess by port if service unknown
        port_service_map = {
            53: "domain_u", 21: "ftp_data", 22: "ssh", 25: "smtp",
            80: "http", 443: "ssl", 8080: "http", 23: "telnet"
        }
        try:
            dst_port_int = int(dst_port) if dst_port else 0
            if dst_port_int in port_service_map:
                service = port_service_map[dst_port_int]
        except:
            pass
    
    # Flag: Better TCP flag mapping
    raw_flags = safe_get(rec, "tcp_flags", "flags") or (rec.get("flow") or {}).get("tcp_flags") or ""
    flag = "SF"  # Default to established connection (normal)
    
    if protocol == "tcp" and raw_flags:
        flag_map = {
            "S": "S0",      # SYN (connection attempt)
            "SA": "SF",     # SYN-ACK (established)
            "A": "SF",      # ACK (data transfer)
            "FA": "RSTO",   # FIN-ACK (closing)
            "RA": "RSTR",   # RST-ACK (reset)
            "PA": "SH",     # PSH-ACK (urgent data)
        }
        flag = flag_map.get(raw_flags, "SF")
    
    # Bytes: Add small random noise to avoid exact zeros
    src_bytes = safe_get(rec, "tx_bytes", "src_bytes") or (rec.get("flow") or {}).get("bytes_toserver", 0)
    dst_bytes = safe_get(rec, "rx_bytes", "dst_bytes") or (rec.get("flow") or {}).get("bytes_toclient", 0)
    
    try:
        src_bytes = int(src_bytes)
        dst_bytes = int(dst_bytes)
    except:
        src_bytes = 0
        dst_bytes = 0
    
    # Add small noise (1-100 bytes) to avoid exact zero
    if src_bytes == 0:
        src_bytes = random.randint(1, 100)
    if dst_bytes == 0:
        dst_bytes = random.randint(1, 100)
    
    duration = safe_get(rec, "duration") or (rec.get("flow") or {}).get("age", 0.0)
    try: 
        duration = float(duration)
    except: 
        duration = 0.0

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

    # BETTER ESTIMATES FOR RATE FEATURES
    same_srv_rate = min(count / 100.0, 1.0) if count > 0 else 0.0
    diff_srv_rate = max(0.0, 1.0 - same_srv_rate)
    
    # BETTER "hot" feature: High traffic = hot
    hot = 0
    if src_bytes > 10000 or dst_bytes > 10000 or count > 10:
        hot = 1
    
    # BETTER "logged_in": Most normal traffic is logged_in=1
    logged_in = 1
    if service in ["eco_i", "ecr_i", "private"] or protocol == "udp":
        logged_in = 0
    
    # Return ALL 35 features with better defaults
    sample = {
        "duration": duration,
        "protocol_type": protocol,  # Now "tcp" not "other"
        "service": service,         # Now "http" not "other"
        "flag": flag,               # Better mapping
        "src_bytes": src_bytes,     # Non-zero with noise
        "dst_bytes": dst_bytes,     # Non-zero with noise
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": hot,                 # Estimated
        "num_failed_logins": 0,
        "logged_in": logged_in,     # Better default
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "is_guest_login": 0,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": 0,
        "srv_serror_rate": 0,
        "rerror_rate": 0,
        "srv_rerror_rate": 0,
        "same_srv_rate": same_srv_rate,      # Estimated
        "diff_srv_rate": diff_srv_rate,      # Estimated
        "srv_diff_host_rate": 0,
        "dst_host_count": count,
        "dst_host_srv_count": srv_count,
        "dst_host_same_srv_rate": same_srv_rate,
        "dst_host_diff_srv_rate": diff_srv_rate,
        "dst_host_same_src_port_rate": 0,
        "dst_host_srv_diff_host_rate": 0,
        "dst_host_serror_rate": 0,
        "dst_host_srv_serror_rate": 0,
        "dst_host_rerror_rate": 0,
        "dst_host_srv_rerror_rate": 0,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "timestamp": ts_str
    }
    return sample



def worker(model,  window, alert_file):
    # Threshold for alerting (can be tuned via env)
    try:
        thresh = float(os.environ.get("IDS_ALERT_THRESHOLD", 0.5))
    except Exception:
        thresh = 0.5
    allow_pred_no_proba = os.environ.get("IDS_ALLOW_PRED_NO_PROBA", "0").lower() in ("1", "true", "yes")

    while True:
        rec = _alert_queue.get()
        try:
            sample = extract_features(rec, window)
            
            # ========== FILTERS ==========
            # FILTER 1: Skip control/management packets with no data
            if (sample["src_bytes"] < 100 and 
                sample["dst_bytes"] < 100 and 
                sample["duration"] == 0.0 and
                sample["count"] < 3):
                continue  
            
            # FILTER 2: Skip DNS queries (usually benign)
            if (sample["service"] == "domain_u" and 
                sample["dst_bytes"] < 500 and 
                sample["duration"] < 0.5):
                continue  
            
            # FILTER 3: Skip very low connection counts (noise)
            if sample["count"] < 2 and sample["srv_count"] < 2:
                continue 
            
            # Use centralized predict() which accepts either a model path or a loaded estimator
            res = predict(model, sample)
            pred = int(res.get("prediction", 0))
            score = res.get("score_attack")

            # ========== DYNAMIC THRESHOLD ==========
            if score is not None:
                # For low-byte traffic, require higher confidence
                if sample["src_bytes"] < 100 and sample["dst_bytes"] < 100:
                    # Require 90% confidence for low-byte traffic
                    lowbyte_thresh = float(os.environ.get("IDS_LOWBYTE_THRESHOLD", "0.9"))
                    is_attack = float(score) >= lowbyte_thresh
                elif sample["protocol_type"] == "other" or sample["service"] == "other":
                    # Require higher confidence for unknown traffic
                    unknown_thresh = float(os.environ.get("IDS_UNKNOWN_THRESHOLD", "0.85"))
                    is_attack = float(score) >= unknown_thresh
                else:
                    # Normal threshold for known traffic patterns
                    is_attack = float(score) >= thresh
            else:
                # if model doesn't expose probability, only treat as attack when allowed
                is_attack = (pred == 1) and allow_pred_no_proba
            
            # Audit log prediction
            try:
                logs_dir = os.path.join(os.getcwd(), 'logs')
                os.makedirs(logs_dir, exist_ok=True)
                with open(os.path.join(logs_dir, 'predictions.log'), 'a', encoding='utf-8') as pl:
                    pl.write(json.dumps({
                        'ts': datetime.now(timezone.utc).isoformat(),
                        'sample': sample,
                        'predict_result': res,
                        'threshold': thresh,
                        'allow_pred_no_proba': allow_pred_no_proba,
                        'is_attack': is_attack
                    }) + "\n")
            except Exception:
                import traceback; traceback.print_exc()

            if is_attack:
                alert = {**sample, "pred": "ATTACK", "ts": sample.get("timestamp") or datetime.now(timezone.utc).isoformat(), "score_attack": score}
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
    print("[i] Start Suricata with: sudo suricata -i <iface> -l /var/log/suricata -D")
    monitor(args.eve)

if __name__ == "__main__":
    main()
