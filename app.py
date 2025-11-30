import os
import sys
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

from flask import Flask, request, jsonify, Response, send_from_directory

from app.models.infer import predict
from app.features.columns_nsl_kdd import COLUMNS, CATEGORICAL, LABEL_COL, DIFFICULTY_COL

try:
    from plyer import notification as plyer_notification
except Exception:
    plyer_notification = None

# -------------------- CONFIG --------------------
MODEL_PATH = os.environ.get("IDS_MODEL_PATH", "models/best_dt.joblib")
EXPECTED_FEATURES = [c for c in COLUMNS if c not in (LABEL_COL, DIFFICULTY_COL)]
SURICATA_LOG = os.environ.get("IDS_SURICATA_LOG", "/var/log/suricata/eve.json")
ALERT_FILE = os.environ.get("IDS_ALERT_FILE", "ids_alerts.jsonl")
NUM_WORKERS = int(os.environ.get("IDS_THREADS", 4))
SLIDING_WINDOW = float(os.environ.get("IDS_WINDOW", 2.0))  # seconds
# ------------------------------------------------

app = Flask(__name__)

_alert_queue = Queue()
_recent_alerts = deque(maxlen=200)
_sse_subscribers = []

# Sliding windows
_by_host = defaultdict(deque)           # dst_ip -> timestamps
_by_host_service = defaultdict(deque)   # (dst_ip, service) -> timestamps


# -------------------- UTILS --------------------
def normalize_sample(sample: dict) -> dict:
    normalized = {}
    for k in EXPECTED_FEATURES:
        if k in sample and sample[k] is not None:
            normalized[k] = sample[k]
        else:
            normalized[k] = "unknown" if k in CATEGORICAL else 0
    return normalized


def extract_features(alert: dict) -> dict:
    """Map Suricata alert/flow JSON to NSL-KDD features."""
    def find(*keys, default=None):
        for k in keys:
            # only test string keys against the alert dict
            if isinstance(k, str) and k in alert:
                return alert[k]
        return default
    # prefer top-level proto, else flow.proto if present
    flow = find("flow", default={}) or {}
    proto = find("proto") or flow.get("proto") or "unknown"
    service = find("app_proto") or find("service") or "other"
    flag = find("tcp_flags") or flow.get("tcp_flags") or "OTH"
    src_bytes = int(find("tx_bytes") or find("src_bytes") or 0)
    dst_bytes = int(find("rx_bytes") or find("dst_bytes") or 0)
    duration = float(find("duration") or 0.0)
    dst_ip = alert.get("dest_ip") or alert.get("dst") or "unknown"

    # Sliding-window counts
    ts = alert.get("timestamp")
    try:
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() if ts else time.time()
    except Exception:
        ts = time.time()

    # Count per host
    dq_host = _by_host[dst_ip]
    dq_host.append(ts)
    while dq_host and ts - dq_host[0] > SLIDING_WINDOW:
        dq_host.popleft()
    count = len(dq_host)

    # Count per host+service
    key_srv = (dst_ip, service)
    dq_srv = _by_host_service[key_srv]
    dq_srv.append(ts)
    while dq_srv and ts - dq_srv[0] > SLIDING_WINDOW:
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
        "src_ip": alert.get("src_ip") or alert.get("src") or "unknown",
        "src_port": alert.get("src_port") or alert.get("sp"),
        "dst_port": alert.get("dest_port") or alert.get("dp"),
        "timestamp": alert.get("timestamp")
    }


def play_alert(sound_path: Optional[str] = None):
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


def broadcast_alert(payload: dict):
    _recent_alerts.appendleft(payload)
    dead = []
    for q in list(_sse_subscribers):
        try:
            q.put_nowait(payload)
        except Exception:
            dead.append(q)
    for d in dead:
        try:
            _sse_subscribers.remove(d)
        except ValueError:
            pass


# -------------------- IDS WORKER --------------------
def ids_worker(pipeline):
    while True:
        alert = _alert_queue.get()
        try:
            sample = extract_features(alert)
            norm = normalize_sample(sample)
            pred = predict(MODEL_PATH, norm)
            is_attack = int(pred.get("prediction", 0)) == 1

            if is_attack:
                record = {
                    "ts": sample["timestamp"] or datetime.now(timezone.utc).isoformat(),
                    "src": sample["src_ip"],
                    "dst": sample["dst_ip"],
                    "sp": sample.get("src_port"),
                    "dp": sample.get("dst_port"),
                    **sample,
                    "pred": "ATTACK"
                }

                # Save alert
                with open(ALERT_FILE, "a", buffering=1) as fh:
                    fh.write(json.dumps(record) + "\n")

                # Broadcast & optional alerts
                broadcast_alert(record)
                sound_path = os.environ.get("IDS_ALERT_SOUND_PATH")
                if os.environ.get("ENABLE_ALERT_SOUND", "0").lower() in ("1", "true", "yes"):
                    play_alert(sound_path)

                if plyer_notification:
                    try:
                        plyer_notification.notify(title="IDS Alert", message="Attack detected", timeout=3)
                    except Exception:
                        logging.exception("Desktop notification failed")

                print(f"[ALERT] {record}")

        except Exception:
            logging.exception("IDS worker error")
        finally:
            _alert_queue.task_done()


# -------------------- SURICATA MONITOR --------------------
def tail_f(path: str):
    while True:
        try:
            with open(path, "r") as f:
                f.seek(0, 2)
                try: inode = os.fstat(f.fileno()).st_ino
                except Exception: inode = None
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
    for line in tail_f(SURICATA_LOG):
        try:
            alert = json.loads(line)
            _alert_queue.put(alert)
        except Exception:
            continue


# -------------------- FLASK ENDPOINTS --------------------
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
    except Exception as exc:
        return jsonify({"error": "Invalid JSON", "details": str(exc)}), 400

    samples = [data] if isinstance(data, dict) else data if isinstance(data, list) else None
    if samples is None:
        return jsonify({"error": "Input must be JSON object or array"}), 400

    results = []
    for sample in samples:
        try:
            norm = normalize_sample(sample)
            res = predict(MODEL_PATH, norm)
            results.append({"input": sample, "prediction": res})
        except Exception as exc:
            results.append({"input": sample, "error": str(exc)})
    return jsonify(results[0] if len(results) == 1 else results)


@app.route('/alerts/stream')
def stream_alerts():
    def event_stream(q: Queue):
        try:
            for a in list(_recent_alerts):
                yield f"data: {json.dumps(a)}\n\n"
            while True:
                try:
                    item = q.get(timeout=15)
                    yield f"data: {json.dumps(item)}\n\n"
                except Empty:
                    yield ": ping\n\n"
        finally:
            try: _sse_subscribers.remove(q)
            except Exception: pass

    q = Queue()
    _sse_subscribers.append(q)
    return Response(event_stream(q), mimetype='text/event-stream')


@app.route('/')
def index():
    return send_from_directory('static', 'dashboard.html')


@app.route('/alerts/download')
def download_alerts():
    try:
        data = list(_recent_alerts)
        ts = time.strftime("%Y%m%d-%H%M%S")
        payload = json.dumps(data, indent=2)
        filename = f"alerts-{ts}.json"
        return Response(payload, mimetype="application/json",
                        headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception:
        logging.exception("Failed to prepare alerts download")
        return jsonify({"error": "Failed to prepare alerts download"}), 500


# -------------------- MAIN --------------------
if __name__ == "__main__":
    try:
        import joblib
        pipeline = joblib.load(MODEL_PATH)
    except Exception:
        app.logger.warning("Model not found; /predict will fail.")

    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=ids_worker, args=(pipeline,), daemon=True)
        t.start()

    threading.Thread(target=monitor_suricata, daemon=True).start()

    # Find free port
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
        raise RuntimeError(f"No free port found in range {start_port}-{start_port+max_tries-1}")

    env_port = os.environ.get("PORT") or os.environ.get("IDS_PORT")
    try:
        desired = int(env_port) if env_port else 5000
    except ValueError:
        desired = 5000

    port = find_free_port(start_port=desired, host="0.0.0.0")
    if port != desired:
        app.logger.warning("Port %s in use; using %s", desired, port)

    app.run(host="0.0.0.0", port=port, debug=True)
