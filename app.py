from flask import Flask, request, jsonify, send_from_directory, Response
from app.models.infer import predict
import os
from flask import Flask, request, jsonify
from app.models.infer import predict
from app.features.columns_nsl_kdd import COLUMNS, CATEGORICAL, LABEL_COL, DIFFICULTY_COL
import os
import logging
import subprocess
import platform
import threading
from collections import deque
from queue import Queue, Empty
from typing import List, Optional
import time
try:
    import simpleaudio as sa
except Exception:
    sa = None
try:
    from plyer import notification as plyer_notification
except Exception:
    plyer_notification = None
import json
import json
import time
import threading
def extract_features(alert: dict) -> dict:
    """Map common Suricata keys to model features. Best-effort."""
    def find(*keys, default=None):
        for k in keys:
            if isinstance(alert, dict) and k in alert:
                return alert[k]
        return default

    proto = find('proto', 'flow', default=None)
    if isinstance(proto, dict):
        proto = proto.get('proto')
    proto = proto or find('packet', 'protocol', default='unknown') or 'unknown'
    service = find('app_proto', 'service', default='unknown') or 'unknown'
    flag = find('tcp_flags', default=None)
    if not flag:
        flow = alert.get('flow') if isinstance(alert.get('flow'), dict) else None
        if flow:
            flag = flow.get('tcp_flags')
    flag = flag or 'unknown'
    src_bytes = find('tx_bytes', 'src_bytes', default=0) or 0
    dst_bytes = find('rx_bytes', 'dst_bytes', default=0) or 0
    return {
        'protocol_type': proto,
        'service': service,
        'flag': flag,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
    }
def monitor_suricata_log(log_path: str, poll_interval: float = 1.0):
    """Background thread: tail Suricata JSONL log and submit new events for prediction."""
    app.logger.info(f"Starting Suricata log monitor: {log_path}")
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            f.seek(0, 2)  # Seek to end
            while True:
                line = f.readline()
                if not line:
                    time.sleep(poll_interval)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    alert = json.loads(line)
                except Exception as e:
                    app.logger.warning(f"Invalid JSON in Suricata log: {e}")
                    continue
                sample = extract_features(alert)
                try:
                    res = predict(MODEL_PATH, normalize_sample(sample))
                    # Broadcast and log as usual
                    broadcast_alert({"input": sample, "prediction": res, "source": "suricata"})
                except Exception as e:
                    app.logger.warning(f"Suricata log prediction failed: {e}")
    except Exception as e:
        app.logger.error(f"Suricata log monitor error: {e}")
import socket

app = Flask(__name__)

# Default model path; can be overridden with environment variable
MODEL_PATH = os.environ.get("IDS_MODEL_PATH", "models/best_dt.joblib")

# Expected features for the model (drop label/difficulty)
EXPECTED_FEATURES = [c for c in COLUMNS if c not in (LABEL_COL, DIFFICULTY_COL)]


def normalize_sample(sample: dict) -> dict:
    """Return a sample dict containing all expected features.

    - For categorical features, missing values are set to 'unknown'.
    - For numeric features, missing values are set to 0.
    """
    if not isinstance(sample, dict):
        raise ValueError("Each sample must be a JSON object (dictionary)")
    normalized = {}
    for k in EXPECTED_FEATURES:
        if k in sample and sample[k] is not None:
            normalized[k] = sample[k]
        else:
            normalized[k] = "unknown" if k in CATEGORICAL else 0
    return normalized


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """POST JSON sample (object) or list of samples. Returns prediction(s).

    Response shape:
    - single input -> single JSON object
    - batch input -> list of JSON objects
    """
    try:
        data = request.get_json(force=True)
    except Exception as exc:
        return jsonify({"error": "Invalid JSON body", "details": str(exc)}), 400

    # normalize incoming payload to a list of samples
    if isinstance(data, dict):
        samples = [data]
    elif isinstance(data, list):
        samples = data
    else:
        return jsonify({"error": "Input must be a JSON object or array of objects"}), 400

    results = []
    for sample in samples:
        try:
            norm = normalize_sample(sample)
            res = predict(MODEL_PATH, norm)
            # If prediction indicates attack (1), optionally play alert sound
            try:
                def _maybe_alert(r):
                    try:
                        if isinstance(r, dict) and int(r.get("prediction", 0)) == 1:
                            enable = os.environ.get("ENABLE_ALERT_SOUND", "0")
                            if enable.lower() in ("1", "true", "yes"):
                                sound_path = os.environ.get("IDS_ALERT_SOUND_PATH")
                                play_alert(sound_path)
                            # send desktop notification if available
                            try:
                                if plyer_notification:
                                    plyer_notification.notify(title="IDS Alert", message="Attack detected by model", timeout=3)
                            except Exception:
                                logging.exception("Desktop notification failed")
                            # broadcast to SSE listeners
                            try:
                                broadcast_alert({"sample": sample, "prediction": r})
                            except Exception:
                                logging.exception("Broadcast failed")
                    except Exception:
                        logging.exception("Alert play failed")

                threading.Thread(target=_maybe_alert, args=(res,), daemon=True).start()
            except Exception:
                logging.exception("Failed to schedule alert")
            results.append({"input": sample, "prediction": res})
        except FileNotFoundError:
            return jsonify({"error": f"Model file not found at '{MODEL_PATH}'. Set IDS_MODEL_PATH or place model there."}), 500
        except ValueError as ve:
            return jsonify({"error": "Invalid sample", "details": str(ve)}), 400
        except Exception as exc:
            logging.exception("Prediction failed")
            return jsonify({"error": "Prediction failed", "details": str(exc)}), 500

    return jsonify(results[0] if len(results) == 1 else results)


def play_alert(sound_path: Optional[str] = None) -> None:
    """Play an alert sound non-blocking.

    - If `sound_path` is provided and the file exists, try platform players:
      - macOS: `afplay`
      - Linux: `aplay`, `paplay`
    - Otherwise emit a terminal bell as a fallback.
    """
    def _play():
        try:
            if sound_path and os.path.exists(sound_path):
                system = platform.system()
                if system == "Darwin":
                    subprocess.run(["afplay", sound_path], check=False)
                    return
                if system == "Linux":
                    # try aplay then paplay
                    for cmd in (["aplay", sound_path], ["paplay", sound_path]):
                        try:
                            subprocess.run(cmd, check=False)
                            return
                        except FileNotFoundError:
                            continue
                # Windows: use winsound if available
                if system == "Windows":
                    try:
                        import winsound
                        winsound.PlaySound(sound_path, winsound.SND_FILENAME)
                        return
                    except Exception:
                        pass
            # Fallback: terminal bell
            print('\a', end='', flush=True)
        except Exception:
            logging.exception("Failed to play alert")

    threading.Thread(target=_play, daemon=True).start()


# Server-sent events (SSE) support: keep simple list of subscribers (Queues)
_sse_subscribers: List[Queue] = []
_recent_alerts = deque(maxlen=200)

def broadcast_alert(payload: dict) -> None:
    """Broadcast an alert payload to SSE subscribers and store in recent deque."""
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


@app.route('/alerts/stream')
def stream_alerts():
    """SSE endpoint that streams alerts as they arrive.

    Clients should connect with `Accept: text/event-stream` and keep the connection open.
    """
    def event_stream(q: Queue):
        try:
            # Send recent alerts first
            for a in list(_recent_alerts):
                yield f"data: {json.dumps(a)}\n\n"
            while True:
                try:
                    item = q.get(timeout=15)
                    yield f"data: {json.dumps(item)}\n\n"
                except Empty:
                    # send a ping comment to keep connection alive
                    yield ": ping\n\n"
        finally:
            try:
                _sse_subscribers.remove(q)
            except Exception:
                pass

    q = Queue()
    _sse_subscribers.append(q)
    return app.response_class(event_stream(q), mimetype='text/event-stream')


@app.route('/')
def index():
    """Serve the demo dashboard as the application homepage."""
    # static/dashboard.html is created by the helper; serve it from the static folder
    return send_from_directory('static', 'dashboard.html')


@app.route('/alerts/download')
def download_alerts():
    """Return recent alerts as a downloadable JSON file.

    The route returns the contents of the in-memory `_recent_alerts` deque
    as a JSON attachment so operators can save the current alerts.
    """
    try:
        data = list(_recent_alerts)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        payload = json.dumps(data, indent=2)
        filename = f"alerts-{timestamp}.json"
        return Response(payload, mimetype="application/json", headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception:
        logging.exception("Failed to prepare alerts download")
        return jsonify({"error": "Failed to prepare alerts download"}), 500


def find_free_port(start_port: int = 5000, host: str = "0.0.0.0", max_tries: int = 100) -> int:
    """Return the first free port starting at `start_port`.

    Attempts up to `max_tries` ports. Raises RuntimeError if none found.
    """
    port = int(start_port)
    for _ in range(max_tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow immediate reuse of address in case of TIME_WAIT
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            s.close()
            return port
        except OSError:
            # port is in use, try next
            try:
                s.close()
            except Exception:
                pass
            port += 1
    raise RuntimeError(f"No free port found in range {start_port}..{start_port + max_tries - 1}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        app.logger.warning("Model file not found at %s. /predict will fail until a model is available.", MODEL_PATH)

    # Always start Suricata log monitor (default path or env override)
    suricata_log = os.environ.get("IDS_SURICATA_LOG", "data/raw/suricata.jsonl")
    t = threading.Thread(target=monitor_suricata_log, args=(suricata_log,), daemon=True)
    t.start()

    # Allow overriding via environment variables: PORT or IDS_PORT
    env_port = os.environ.get("PORT") or os.environ.get("IDS_PORT")
    try:
        desired = int(env_port) if env_port is not None else 5000
    except ValueError:
        desired = 5000

    try:
        port = find_free_port(start_port=desired, host="0.0.0.0", max_tries=200)
    except RuntimeError as rexc:
        app.logger.exception("Failed to find a free port: %s", rexc)
        raise

    if port != int(desired):
        app.logger.warning("Port %s was in use; using next free port %s", desired, port)
    else:
        app.logger.info("Using port %s", port)

    app.run(host="0.0.0.0", port=port, debug=True)
