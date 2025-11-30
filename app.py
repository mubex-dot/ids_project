import os
import platform
import subprocess
import threading
import logging
import json
import time
from collections import deque
from queue import Queue, Empty
from typing import Optional, List
from flask import Flask, request, jsonify, send_from_directory, Response
from app.models.infer import predict
from app.features.columns_nsl_kdd import CATEGORICAL  # just for normalization

# === CONFIG ===
MODEL_PATH = os.environ.get("IDS_MODEL_PATH", "models/best_dt.joblib")
EXPECTED_FEATURES = ["protocol_type", "service", "flag", "src_bytes", "dst_bytes"]

app = Flask(__name__)

# SSE subscribers
_sse_subscribers: List[Queue] = []
_recent_alerts = deque(maxlen=200)

# --- Utility Functions ---
def normalize_sample(sample: dict) -> dict:
    """Ensure all required features exist and are valid."""
    normalized = {}
    for f in EXPECTED_FEATURES:
        normalized[f] = sample.get(f, "unknown" if f in CATEGORICAL else 0)
    return normalized

def play_alert(sound_path: Optional[str] = None) -> None:
    """Play alert sound non-blocking."""
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
                    return
            print('\a', end='', flush=True)  # fallback terminal bell
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

# --- Routes ---
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        samples = [data] if isinstance(data, dict) else data
        results = []
        for s in samples:
            norm = normalize_sample(s)
            res = predict(MODEL_PATH, norm)
            if int(res.get("prediction", 0)) == 1:
                # Alert triggers
                sound_path = os.environ.get("IDS_ALERT_SOUND_PATH")
                play_alert(sound_path)
                try:
                    from plyer import notification
                    notification.notify(title="IDS Alert", message="Attack detected", timeout=3)
                except Exception:
                    pass
                broadcast_alert({"sample": norm, "prediction": res})
            results.append({"input": s, "prediction": res})
        return jsonify(results[0] if len(results) == 1 else results)
    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route('/alerts/stream')
def stream_alerts():
    def event_stream(q: Queue):
        for a in list(_recent_alerts):
            yield f"data: {json.dumps(a)}\n\n"
        while True:
            try:
                item = q.get(timeout=15)
                yield f"data: {json.dumps(item)}\n\n"
            except Empty:
                yield ": ping\n\n"
    q = Queue()
    _sse_subscribers.append(q)
    return app.response_class(event_stream(q), mimetype='text/event-stream')

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
        return jsonify({"error": "Failed"}), 500

# --- Suricata Log Monitor ---
def extract_features(alert: dict) -> dict:
    """Extract only required features from Suricata JSON."""
    return {
        "protocol_type": alert.get("proto", "unknown"),
        "service": alert.get("app_proto", "unknown"),
        "flag": alert.get("tcp_flags", "OTH"),
        "src_bytes": alert.get("tx_bytes", 0),
        "dst_bytes": alert.get("rx_bytes", 0),
    }

def monitor_suricata_log(log_path: str, poll_interval: float = 1.0):
    app.logger.info(f"Monitoring Suricata log: {log_path}")
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            f.seek(0, 2)  # go to end
            while True:
                line = f.readline()
                if not line:
                    time.sleep(poll_interval)
                    continue
                try:
                    alert = json.loads(line.strip())
                    sample = extract_features(alert)
                    norm = normalize_sample(sample)
                    res = predict(MODEL_PATH, norm)
                    if int(res.get("prediction", 0)) == 1:
                        broadcast_alert({"sample": norm, "prediction": res})
                except Exception:
                    logging.exception("Failed processing Suricata line")
    except Exception:
        logging.exception("Suricata log monitoring failed")

# --- Run Flask ---
if __name__ == "__main__":
    # Start Suricata monitor
    log_file = os.environ.get("IDS_SURICATA_LOG", "/var/log/suricata/eve.json")
    threading.Thread(target=monitor_suricata_log, args=(log_file,), daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("IDS_PORT", 5000)), debug=True)
