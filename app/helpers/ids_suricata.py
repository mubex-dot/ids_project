
"""
NOTE:
- Tails Suricata's eve.json and classifies each flow (event_type=flow) using a saved sklearn Pipeline.
- Designed for IDS models trained on NSL-KDD-like features (protocol_type, service, flag, duration, src_bytes, dst_bytes, count, srv_count).
- Start Suricata first: sudo suricata -i <iface> -l /var/log/suricata -D
- Then run: sudo python3 live_ids_suricata.py --model models/best_dt.joblib --eve /var/log/suricata/eve.json

NOTE:
- The pipeline should use OneHotEncoder(handle_unknown="ignore") for categorical columns.
- We approximate KDD flags from Suricata flow/tcp state.
- We keep 2-second sliding windows per dst host and per (dst, service) to compute count and srv_count.
"""

import argparse
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
import os
import sys

import joblib
import pandas as pd

# --- basic port->service mapping to approximate NSL-KDD 'service' ---
PORT_SERVICE = {
    20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'domain_u',
    67: 'dhcp', 68: 'dhcp', 69: 'tftp_u', 80: 'http', 110: 'pop_3', 111: 'rpc', 113: 'auth',
    119: 'nntp', 123: 'ntp_u', 135: 'msrpc', 137: 'netbios_ns', 138: 'netbios_dgm',
    139: 'netbios_ssn', 143: 'imap4', 161: 'snmp', 162: 'snmp', 179: 'bgp', 389: 'ldap',
    443: 'http', 445: 'microsoft_ds', 512: 'exec', 513: 'login', 514: 'shell',
    515: 'printer', 520: 'efs', 540: 'uucp', 548: 'afp', 554: 'rtsp', 587: 'submission',
    631: 'cups', 993: 'imap', 995: 'pop3', 1080: 'socks', 1433: 'sql_net',
    1521: 'oracle', 2049: 'nfs', 2082: 'other', 2083: 'other', 2222: 'other',
    3306: 'mysql', 3389: 'remote_login', 3690: 'svn', 4333: 'https_443', 5000: 'other',
    5432: 'postgres', 5900: 'vnc', 6379: 'redis', 8000: 'http_8001', 8001: 'http_8001',
    8080: 'http', 8443: 'http', 9000: 'other', 9200: 'other'
}

def port_to_service(port: int) -> str:
    return PORT_SERVICE.get(int(port), 'other')

def proto_to_protocol_type(proto: str) -> str:
    p = (proto or '').lower()
    if p in ('tcp', 'udp', 'icmp'):
        return p
    return 'other'  # OneHotEncoder(handle_unknown="ignore") should swallow this

def suri_state_to_flag(suri_state: str) -> str:
    """
    Map Suricata flow/tcp state to a coarse KDD-like 'flag' string.
    This is an approximation good enough to drive the categorical pipeline.
    """
    if not suri_state:
        return 'OTH'
    s = suri_state.lower()
    if 'syn' in s and 'ack' not in s:
        return 'S0'   # SYN seen, no response
    if 'established' in s or 'fin' in s or 'closed' in s:
        return 'SF'   # normal completed connections
    if 'rst' in s:
        return 'REJ'  # reset
    return 'OTH'

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--model',
        default='models/best_dt.joblib',
        help='Path to saved sklearn Pipeline (joblib)'
    )
    ap.add_argument(
        '--eve',
        default='/var/log/suricata/eve.json',
        help='Path to Suricata eve.json'
    )
    ap.add_argument('--window', type=float, default=2.0, help='Seconds for count/srv_count window')
    ap.add_argument('--print-cols', action='store_true', help='Print expected model input columns and exit')
    ap.add_argument('--alert-file', default='ids_alerts.jsonl', help='Write alerts to this JSONL file')
    return ap.parse_args()


def get_expected_columns(pipeline):
    # Try to introspect the first ColumnTransformer in the pipeline
    expected = []
    try:
        for name, step in pipeline.named_steps.items():
            if hasattr(step, 'transformers_'):
                for _, _, cols in step.transformers_:
                    if isinstance(cols, (list, tuple)):
                        expected.extend(cols)
                break
    except Exception:
        pass
    return list(dict.fromkeys(expected))

def tail_f(path):
    # Like `tail -F`: follow file as it grows; handle rotations by reopening
    with open(path, 'r') as f:
        f.seek(0, os.SEEK_END)
        inode = os.fstat(f.fileno()).st_ino
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(0.2)
                try:
                    if os.stat(path).st_ino != inode:
                        # rotated
                        f.close()
                        f = open(path, 'r')
                        inode = os.fstat(f.fileno()).st_ino
                except FileNotFoundError:
                    time.sleep(0.5)

def main():
    args = parse_args()
    print(f"[+] Loading model: {args.model}")
    pipeline = joblib.load(args.model)

    expected_cols = get_expected_columns(pipeline)
    if args.print_cols:
        print("Expected input columns:", expected_cols)
        sys.exit(0)

    print(f"[+] Tail Suricata eve: {args.eve}")
    print("[i] Start Suricata with: sudo suricata -i <iface> -l /var/log/suricata -D")
    print("[i] Press Ctrl+C to stop.")

    # Sliding windows for count/srv_count
    by_host = defaultdict(deque)         # dst_ip -> deque[timestamps]
    by_host_srv = defaultdict(deque)     # (dst_ip, service) -> deque[timestamps]

    # Open alert output
    alert_fh = open(args.alert_file, 'a', buffering=1)

    for raw in tail_f(args.eve):
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        if rec.get('event_type') != 'flow':
            continue

        flow = rec.get('flow', {})
        src = rec.get('src_ip')
        dst = rec.get('dest_ip')
        sp = rec.get('src_port')
        dp = rec.get('dest_port')
        proto = rec.get('proto')
        state = flow.get('state') or (rec.get('tcp', {}) or {}).get('state')

        # timestamps
        ts_str = rec.get('timestamp')
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp() if ts_str else time.time()
        except Exception:
            ts = time.time()

        # Compute features
        protocol_type = proto_to_protocol_type(proto)
        service = port_to_service(dp) if dp is not None else 'other'
        flag = suri_state_to_flag(state)

        # Duration
        # Suricata flow may include "start" and "end"/"age" or "duration"
        start = flow.get('start')
        end = flow.get('end')
        age = flow.get('age')
        if start and end:
            try:
                t0 = datetime.fromisoformat(start.replace('Z','+00:00')).timestamp()
                t1 = datetime.fromisoformat(end.replace('Z','+00:00')).timestamp()
                duration = max(0.0, t1 - t0)
            except Exception:
                duration = float(age) if age is not None else 0.0
        else:
            duration = float(age) if age is not None else 0.0

        # Bytes
        src_bytes = int(flow.get('bytes_toserver', 0))
        dst_bytes = int(flow.get('bytes_toclient', 0))

        # Sliding-window counts
        win = args.window
        dq_h = by_host[dst]
        dq_h.append(ts)
        while dq_h and ts - dq_h[0] > win:
            dq_h.popleft()
        count = len(dq_h)

        key_srv = (dst, service)
        dq_hs = by_host_srv[key_srv]
        dq_hs.append(ts)
        while dq_hs and ts - dq_hs[0] > win:
            dq_hs.popleft()
        srv_count = len(dq_hs)

        # Build input row with expected columns (fill defaults)
        row = {}
        for c in expected_cols or ['protocol_type','service','flag','duration','src_bytes','dst_bytes','count','srv_count']:
            if c == 'protocol_type':
                row[c] = protocol_type
            elif c == 'service':
                row[c] = service
            elif c == 'flag':
                row[c] = flag
            elif c == 'duration':
                row[c] = duration
            elif c == 'src_bytes':
                row[c] = src_bytes
            elif c == 'dst_bytes':
                row[c] = dst_bytes
            elif c == 'count':
                row[c] = count
            elif c == 'srv_count':
                row[c] = srv_count
            else:
                # Unknown column: put a benign default (numeric 0 / categorical 'other')
                row[c] = 0

        df = pd.DataFrame([row])
        try:
            pred = int(pipeline.predict(df)[0])
            if pred == 1:
                alert = {
                    "ts": ts_str or datetime.now(timezone.utc).isoformat(),
                    "src": src, "dst": dst, "sp": sp, "dp": dp,
                    "proto": protocol_type, "service": service, "flag": flag,
                    "duration": duration, "src_bytes": src_bytes, "dst_bytes": dst_bytes,
                    "count": count, "srv_count": srv_count,
                    "pred": "ATTACK"
                }
                print(f"[ALERT] {alert}")
                alert_fh.write(json.dumps(alert) + "\n")
        except Exception as e:
            # Model threw due to unknown columns? Print once then continue.
            sys.stderr.write(f"[!] Prediction error: {e}\n")
            time.sleep(0.2)

if __name__ == "__main__":
    main()
