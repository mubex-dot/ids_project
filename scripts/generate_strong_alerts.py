#!/usr/bin/env python3
"""Generate and optionally POST strong attack samples for demo/testing.

Writes newline-delimited JSON to a file (default: data/raw/suricata.jsonl) or POSTs each sample
as a single /predict call to a running IDS API.

Usage examples:
  # write 10 strong alerts to the default file
  python scripts/generate_strong_alerts.py --count 10

  # post 5 strong alerts to the running API
  python scripts/generate_strong_alerts.py --count 5 --post-url http://127.0.0.1:5001/predict

Options:
  --strength  (1..3)  how "strong" the attack signal should be (3 strongest)
"""
import argparse
import json
import random
import time
import sys
from datetime import datetime


def make_strong_attack(i: int, strength: int = 2):
    # Base values, escalate with `strength`
    mult = {1: 10, 2: 100, 3: 1000}.get(strength, 100)
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "proto": "tcp",
        "app_proto": "http",
        "tcp_flags": random.choice(["S", "F", "R"]),
        "tx_bytes": random.randint(1000 * mult, 1000000 * mult),
        "rx_bytes": random.randint(0, 10),
        # full NSL-KDD style numeric features (set to aggressive values)
        "duration": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 20 * strength,
        "num_failed_logins": 5 * strength,
        "logged_in": 0,
        "num_compromised": 1 * strength,
        "root_shell": 1 if strength >= 2 else 0,
        "su_attempted": 0,
        "num_root": 0,
        "is_guest_login": 0,
        "count": 1000 * strength,
        "srv_count": 500 * strength,
        "serror_rate": 1.0,
        "srv_serror_rate": 1.0,
        "rerror_rate": 1.0,
        "srv_rerror_rate": 1.0,
        "same_srv_rate": 0.0,
        "diff_srv_rate": 1.0,
        "srv_diff_host_rate": 1.0,
        "dst_host_count": 255,
        "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 0.0,
        "dst_host_diff_srv_rate": 1.0,
        "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 1.0,
        "dst_host_serror_rate": 1.0,
        "dst_host_srv_serror_rate": 1.0,
        "dst_host_rerror_rate": 1.0,
        "dst_host_srv_rerror_rate": 1.0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "S",
        "demo_id": f"strong-{i}-s{strength}",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", "-c", type=int, default=5)
    p.add_argument("--interval", "-i", type=float, default=0.5)
    p.add_argument("--output", "-o", default="data/raw/suricata.jsonl")
    p.add_argument("--post-url", help="If provided, POST each sample to this URL instead of writing to file")
    p.add_argument("--strength", "-s", type=int, choices=[1, 2, 3], default=2)
    args = p.parse_args()

    if args.post_url:
        import requests

    # ensure output dir exists if writing file
    if not args.post_url:
        import os
        d = os.path.dirname(args.output)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    for i in range(1, args.count + 1):
        sample = make_strong_attack(i, args.strength)
        if args.post_url:
            try:
                resp = requests.post(args.post_url, json=sample, timeout=5)
                print(i, resp.status_code, resp.text)
            except Exception as e:
                print('post failed', e, file=sys.stderr)
        else:
            with open(args.output, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(sample) + "\n")
            print('wrote', i, sample.get('demo_id'))
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
