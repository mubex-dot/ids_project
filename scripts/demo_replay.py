import argparse
import json
import random
import time
import os
from datetime import datetime


def make_benign(i: int):
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "proto": "tcp",
        "app_proto": "http",
        "tcp_flags": "SF",
        "tx_bytes": int(random.gauss(200, 50)),
        "rx_bytes": int(random.gauss(80, 30)),
        "demo_id": f"benign-{i}",
    }


def make_attack(i: int):
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "proto": "tcp",
        "app_proto": "http",
        "tcp_flags": random.choice(["S", "R", "F"]),
        "tx_bytes": random.randint(50000, 1000000),
        "rx_bytes": random.randint(0, 10),
        "demo_id": f"attack-{i}",
    }


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def replay(path: str, count: int, interval: float, attack_ratio: float = 0.25):
    ensure_dir(path)
    print(f"Replaying {count} events to {path} (interval={interval}s, attack_ratio={attack_ratio})")
    with open(path, "a", encoding="utf-8") as fh:
        for i in range(1, count + 1):
            is_attack = random.random() < attack_ratio
            if is_attack:
                evt = make_attack(i)
            else:
                evt = make_benign(i)
            line = json.dumps(evt, separators=(",", ":"))
            fh.write(line + "\n")
            fh.flush()
            print("Wrote:", line)
            time.sleep(interval)


def main():
    p = argparse.ArgumentParser(description="Replay demo Suricata JSONL alerts")
    p.add_argument("--output", "-o", default="data/raw/suricata.jsonl", help="Output JSONL path")
    p.add_argument("--count", "-c", type=int, default=10, help="Number of events to write")
    p.add_argument("--interval", "-i", type=float, default=0.5, help="Seconds between events")
    p.add_argument("--attack-ratio", "-r", type=float, default=0.25, help="Fraction of events that are attacks (0..1)")
    args = p.parse_args()

    try:
        replay(args.output, args.count, args.interval, args.attack_ratio)
    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == "__main__":
    main()
