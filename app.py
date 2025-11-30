
import os
import json
import time
import threading
import logging
from collections import deque, defaultdict
from queue import Queue, Empty
from datetime import datetime, timezone
from typing import Optional
import platform
import subprocess
import socket

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO

# Try to use user's predict helper if present, else fall back to joblib
USE_APP_PREDICT = False
try:
    from app.models.infer import predict as app_predict
    USE_APP_PREDICT = True
except Exception:
    app_predict = None

try:
    import joblib
except Exception:
    joblib = None

# -------------------- CONFIG --------------------
MODEL_PATH = os.environ.get("IDS_MODEL_PATH", "models/best_dt.joblib")
SURICATA_LOG = os.environ.get("IDS_SURICATA_LOG", "/var/log/suricata/eve.json")
ALERT_FILE = os.environ.get("IDS_ALERT_FILE", "ids_alerts.jsonl")
NUM_WORKERS = int(os.environ.get("IDS_THREADS", 4))
SLIDING_WINDOW = float(os.environ.get("IDS_WINDOW", 2.0))  # seconds
ENABLE_SOUND = os.environ.get("ENABLE_ALERT_SOUND", "0").lower() in ("1", "true", "yes")
SOUND_PATH = os.environ.get("IDS_ALERT_SOUND_PATH", "static/alert.wav")
# ------------------------------------------------

app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

_alert_queue = Queue()
_recent_alerts = deque(maxlen=500)
_sse_subscribers = []

# sliding-window structures (shared)
_by_host = defaultdict(deque)           # dst_ip -> deque[timestamps]
_by_host_service = defaultdict(deque)   # (dst_ip, service) -> deque[timestamps]

# attempt to load joblib pipeline if available
PIPELINE = None
if not USE_APP_PREDICT and joblib:
    try:
        PIPELINE = joblib.load(MODEL_PATH)
        app.logger.info(f"Loaded joblib pipeline from {MODEL_PATH}")
    except Exception as e:
        PIPELINE = None
        app.logger.warning(f"Could not load pipeline at {MODEL_PATH}: {e}")

# -------------------- UTILITIES --------------------
def play_alert(sound_path: Optional[str] = None):
    def _play():
        try:
            sp = sound_path or SOUND_PATH
            if sp and os.path.exists(sp):
                system = platform.system()
                if system == "Darwin":
                    subprocess.run(["afplay", sp], check=False)
                elif system == "Linux":
                    for cmd in (["aplay", sp], ["paplay", sp]):
                        try:
                            subprocess.run(cmd, check=False)
                            return
                        except FileNotFoundError:
                            continue
                elif system == "Windows":
                    import winsound
                    winsound.PlaySound(sp, winsound.SND_FILENAME)
                return
            # fallback: terminal bell
            print('\a', end='', flush=True)
        except Exception:
            app.logger.exception("Alert sound failed")
    threading.Thread(target=_play, daemon=True).start()

def broadcast_socketio(alert: dict):
    """Send to web clients via SocketIO and store locally."""
    _recent_alerts.appendleft(alert)
    socketio.emit("alert", alert)

def safe_get(d, *keys, default=None):
    """Return first found top-level key in dict d or default."""
    for k in keys:
        if isinstance(k, str) and k in d:
            return d[k]
    return default

# -------------------- FEATURE EXTRACTION (fixed) --------------------
def extract_features(rec: dict) -> dict:
    """
    Safely extract NSL-KDD-like features from a Suricata JSON record.
    Returns a dict with model features and metadata (src/dst/timestamp).
    """
    # top-level helpers
    src_ip = safe_get(rec, "src_ip", "src", default=None)
    dst_ip = safe_get(rec, "dest_ip", "dst", default=None)
    src_port = safe_get(rec, "src_port", "sp", default=None)
    dst_port = safe_get(rec, "dest_port", "dp", default=None)

    # protocol/service/flags (nested safe access)
    protocol = safe_get(rec, "proto", "protocol", default=None)
    if not protocol:
        protocol = (rec.get("flow") or {}).get("proto") or (rec.get("packet") or {}).get("protocol")

    service = safe_get(rec, "app_proto", "service", default=None)
    if not service:
        service = (rec.get("flow") or {}).get("service")

    flag = safe_get(rec, "tcp_flags", "flags", default=None)
    if not flag:
        flag = (rec.get("flow") or {}).get("tcp_flags") or (rec.get("tcp") or {}).get("state")

    # bytes/duration
    src_bytes = safe_get(rec, "tx_bytes", "src_bytes", default=None)
    if src_bytes is None:
        src_bytes = (rec.get("flow") or {}).get("bytes_toserver", 0)
    dst_bytes = safe_get(rec, "rx_bytes", "dst_bytes", default=None)
    if dst_bytes is None:
        dst_bytes = (rec.get("flow") or {}).get("bytes_toclient", 0)

    duration = safe_get(rec, "duration", default=None)
    if duration is None:
        duration = (rec.get("flow") or {}).get("age", 0.0)

    # timestamp -> numeric
    ts_str = safe_get(rec, "timestamp", "ts", default=None)
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() if ts_str else time.time()
    except Exception:
        ts = time.time()

    # sliding-window counts (update global structures)
    dst = dst_ip or "unknown"
    svc = service or "other"
    dq = _by_host[dst]
    dq.append(ts)
    while dq and ts - dq[0] > SLIDING_WINDOW:
        dq.popleft()
    count = len(dq)

    dq2 = _by_host_service[(dst, svc)]
    dq2.append(ts)
    while dq2 and ts - dq2[0] > SLIDING_WINDOW:
        dq2.popleft()
    srv_count = len(dq2)

    # normalize types & defaults
    protocol_type = str(protocol).lower() if protocol else "other"
    service = str(service) if service else "other"
    flag = str(flag) if flag else "OTH"
    try:
        src_bytes = int(src_bytes or 0)
    except Exception:
        src_bytes = 0
    try:
        dst_bytes = int(dst_bytes or 0)
    except Exception:
        dst_bytes = 0
    try:
        duration = float(duration or 0.0)
    except Exception:
        duration = 0.0

    sample = {
        "protocol_type": protocol_type,
        "service": service,
        "flag": flag,
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "count": count,
        "srv_count": srv_count,
        # metadata
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "timestamp": ts_str or datetime.now(timezone.utc).isoformat()
    }
    return sample

# -------------------- PREDICTION PIPELINE --------------------
def predict_sample(sample: dict):
    """
    Predict using user's app.models.infer.predict (if available) or joblib pipeline.
    Returns: a dict like {"prediction": 1} or {"prediction": 0} or raises.
    """
    if USE_APP_PREDICT:
        # app_predict signature assumed: predict(model_path, sample) -> dict or value
        try:
            return app_predict(MODEL_PATH, sample)
        except Exception as e:
            raise
    else:
        if PIPELINE is None:
            raise RuntimeError("No pipeline available (joblib not loaded and app.predict not available)")
        # pipeline expects a row-like mapping; many pipelines accept DataFrame
        import pandas as pd
        df = pd.DataFrame([sample])
        pred = PIPELINE.predict(df)[0]
        # If pipeline returns numeric/class label, normalize to dict
        return {"prediction": int(pred)} if isinstance(pred, (int, float)) else {"prediction": pred}

# -------------------- WORKER --------------------
def ids_worker():
    while True:
        rec = _alert_queue.get()
        try:
            sample = extract_features(rec)
            # normalize sample keys for model if necessary (user pipeline should include an encoder)
            res = predict_sample(sample)
            # Expect res to be dict or scalar
            pred_val = res.get("prediction") if isinstance(res, dict) else res
            is_attack = int(pred_val) == 1
            if is_attack:
                alert = {
                    "ts": sample["timestamp"],
                    "src": sample["src_ip"],
                    "dst": sample["dst_ip"],
                    "src_port": sample["src_port"],
                    "dst_port": sample["dst_port"],
                    "protocol_type": sample["protocol_type"],
                    "service": sample["service"],
                    "flag": sample["flag"],
                    "duration": sample["duration"],
                    "src_bytes": sample["src_bytes"],
                    "dst_bytes": sample["dst_bytes"],
                    "count": sample["count"],
                    "srv_count": sample["srv_count"],
                    "pred": "ATTACK"
                }
                # log to disk
                try:
                    with open(ALERT_FILE, "a", buffering=1) as fh:
                        fh.write(json.dumps(alert) + "\n")
                except Exception:
                    app.logger.exception("Failed writing alert file")
                # broadcast
                broadcast_socketio(alert)
                # play sound
                if ENABLE_SOUND:
                    play_alert(SOUND_PATH)
                app.logger.warning(f"[ALERT] {alert}")
        except Exception:
            app.logger.exception("IDS worker error")
        finally:
            _alert_queue.task_done()

# -------------------- SURICATA LOG TAIL --------------------
def tail_f(path: str):
    """Follow file, handle rotation."""
    while True:
        try:
            with open(path, "r", encoding="utf-8") as f:
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

def monitor_suricata():
    app.logger.info(f"Monitoring Suricata log: {SURICATA_LOG}")
    for line in tail_f(SURICATA_LOG):
        try:
            rec = json.loads(line)
            _alert_queue.put(rec)
        except Exception:
            # ignore parse errors
            continue

# -------------------- FLASK ROUTES --------------------
@app.route("/")
def index():
    return send_from_directory("static", "dashboard.html")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)
    if isinstance(data, dict):
        samples = [data]
    elif isinstance(data, list):
        samples = data
    else:
        return jsonify({"error": "Input must be JSON object or list"}), 400

    results = []
    for s in samples:
        try:
            sample = extract_features(s) if "event_type" in s or "src_ip" in s else s
            res = predict_sample(sample)
            results.append({"input": s, "prediction": res})
        except Exception as e:
            results.append({"input": s, "error": str(e)})
    return jsonify(results[0] if len(results) == 1 else results)

@app.route("/alerts/download")
def download_alerts():
    return jsonify(list(_recent_alerts))

# -------------------- SOCKETIO CLIENTS (for debug) --------------------
@socketio.on("connect")
def on_connect():
    app.logger.info("Web client connected")
    # send recent alerts
    for a in list(_recent_alerts)[:100]:
        socketio.emit("alert", a)

# -------------------- STARTUP --------------------
def find_free_port(start_port: int = 5000, host: str = "0.0.0.0", max_tries: int = 100) -> int:
    port = int(start_port)
    for _ in range(max_tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            s.close()
            return port
        except OSError:
            port += 1
    raise RuntimeError("No free port found")

if __name__ == "__main__":
    # start worker threads
    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=ids_worker, daemon=True)
        t.start()

    # start monitor thread
    threading.Thread(target=monitor_suricata, daemon=True).start()

    # run Flask + SocketIO
    desired = int(os.environ.get("PORT", 5000))
    port = find_free_port(start_port=desired)
    if port != desired:
        app.logger.warning("Port %s in use; using %s", desired, port)
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
