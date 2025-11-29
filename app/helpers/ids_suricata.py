"""
Standalone IDS that tails Suricata eve.json and classifies flows using a saved sklearn Pipeline.
Supports sliding-window counts for host/service, multithreading, and JSONL alert output.

Usage:
    sudo python3 live_ids_suricata.py --model models/best_dt.joblib --eve /var/log/suricata/eve.json
"""

import argparse
import json
import time
import threading
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone
from queue import Queue, Empty
import os
import platform
import subprocess

import joblib
import pandas as pd

# -------------------- CONFIG --------------------
NUM_WORKERS = 4
SLIDING_WINDOW = 2.0  # seconds
ALERT_FILE = "ids_alerts.jsonl"
# --------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best_dt.joblib", help="Path to saved sklearn Pipeline (joblib)")
    ap.add_argument("--eve", default="data/raw/ids_alerts.jsonl", help="Path to Suricata eve.json")
    ap.add_argument("--window", type=float, default=SLIDING_WINDOW, help="Sliding window seconds")
    ap.add_argument("--alert-file", default=ALERT_FILE, help="Write alerts to this JSONL file")
    return ap.parse_args()


# -------------------- SLIDING WINDOWS --------------------
_by_host = defaultdict(deque)
_by_host_service = defaultdict(deque)


# -------------------- ALERT QUEUE --------------------
_alert_queue = Queue()
_recent_alerts = deque(maxlen=200)


# -------------------- UTILS --------------------
def normalize_sample(sample: dict, expected_cols: list):
    normalized = {}
    for c in expected_cols:
        if c in sample and sample[c] is not None:
            normalized[c] = sample[c]
        else:
            normalized[c] = "unknown" if c in ['protocol_type', 'service', 'flag'] else 0
    return normalized


def play_alert(sound_path: str = None):
    def _play():
        try:
            if sound_path and os.path.exists(sound_path):
                system = platform.system()
                if system == "Darwin":
                    subprocess.run(["afplay", sound_path], check=False)
                elif system == "Linux":
                    for cmd in (["aplay", sound_path], ["paplay", sound_path]):
                        try:
                            subprocess.run(cmd, check=False)
                            return
                        except FileNotFoundError:
                            continue
                elif system == "Windows":
                    import winsound
                    winsound.PlaySound(sound_path, winsound.SND_FILENAME)
            else:
                print('\a', end='', flush=True)
        except Exception:
            logging.exception("Failed to play alert")
    threading.Thread(target=_play, daemon=True).start()


# -------------------- IDS WORKER --------------------
def ids_worker(pipeline, expected_cols, window, alert_file):
    while True:
        alert = _alert_queue.get()
        try:
            sample = extract_features(alert, window)
            norm = normalize_sample(sample, expected_cols)
            df = pd.DataFrame([norm])
            pred = int(pipeline.predict(df)[0])
            if pred == 1:
                record = {**sample, "pred": "ATTACK", "ts": sample.get("timestamp") or datetime.now(timezone.utc).isoformat()}
                print(f"[ALERT] {record}")
                _recent_alerts.appendleft(record)
                with open(alert_file, "a", buffering=1) as fh:
                    fh.write(json.dumps(record) + "\n")
                sound_path = os.environ.get("IDS_ALERT_SOUND_PATH")
                if os.environ.get("ENABLE_ALERT_SOUND", "0").lower() in ("1", "true", "yes"):
                    play_alert(sound_path)
        except Exception:
            logging.exception("IDS worker error")
        finally:
            _alert_queue.task_done()


# -------------------- FEATURE EXTRACTION --------------------
def extract_features(alert: dict, window: float):
    ts_str = alert.get("timestamp")
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() if ts_str else time.time()
    except Exception:
        ts = time.time()

    proto = alert.get("proto") or alert.get("protocol_type") or "unknown"
    service = alert.get("service") or alert.get("app_proto") or "other"
    flag = alert.get("tcp_flags") or alert.get("flag") or "OTH"
    src_bytes = int(alert.get("src_bytes") or alert.get("tx_bytes") or 0)
    dst_bytes = int(alert.get("dst_bytes") or alert.get("rx_bytes") or 0)
    duration = float(alert.get("duration") or 0.0)
    dst_ip = alert.get("dest_ip") or alert.get("dst") or "unknown"
    src_ip = alert.get("src_ip") or alert.get("src") or "unknown"
    src_port = alert.get("src_port") or alert.get("sp")
    dst_port = alert.get("dest_port") or alert.get("dp")

    # Sliding-window counts
    dq_host = _by_host[dst_ip]
    dq_host.append(ts)
    while dq_host and ts - dq_host[0] > window:
        dq_host.popleft()
    count = len(dq_host)

    dq_srv = _by_host_service[(dst_ip, service)]
    dq_srv.append(ts)
    while dq_srv and ts - dq_srv[0] > window:
        dq_srv.popleft()
    srv_count = len(dq_srv)

    return {
        "protocol_type": proto,
        "service": service,
        "flag": flag,
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "count": count,
        "srv_count": srv_count,
        "dst_ip": dst_ip,
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "timestamp": ts_str
    }


# -------------------- TAIL SURICATA --------------------
def tail_f(path):
    while True:
        try:
            with open(path, "r") as f:
                f.seek(0, 2)
                try:
                    inode = os.fstat(f.fileno()).st_ino
                except Exception:
                    inode = None
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


def monitor_suricata(eve_path):
    for line in tail_f(eve_path):
        try:
            alert = json.loads(line)
            _alert_queue.put(alert)
        except Exception:
            continue


# -------------------- MAIN --------------------
def main():
    args = parse_args()
    print(f"[+] Loading model: {args.model}")
    pipeline = joblib.load(args.model)

    # Determine expected columns
    try:
        expected_cols = []
        for name, step in pipeline.named_steps.items():
            if hasattr(step, "transformers_"):
                for _, _, cols in step.transformers_:
                    if isinstance(cols, (list, tuple)):
                        expected_cols.extend(cols)
                break
    except Exception:
        expected_cols = ["protocol_type","service","flag","duration","src_bytes","dst_bytes","count","srv_count"]

    # Start workers
    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=ids_worker, args=(pipeline, expected_cols, args.window, args.alert_file), daemon=True)
        t.start()

    # Start Suricata monitor
    print(f"[+] Monitoring Suricata log: {args.eve}")
    monitor_suricata(args.eve)


if __name__ == "__main__":
    main()
