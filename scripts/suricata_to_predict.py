#!/usr/bin/env python3
"""
Read Suricata newline-delimited JSON alerts and send them for prediction.

Usage examples:
  # Call local predict() directly (requires project environment and model file)
  python scripts/suricata_to_predict.py --input alerts.jsonl --model models/best_dt.joblib

  # POST to a running Flask API
  python scripts/suricata_to_predict.py --input alerts.jsonl --url http://127.0.0.1:5000/predict

The script tries to extract a minimal set of features from each alert and maps them to
`protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`. You can adapt `extract_features`
if your suricata schema differs.
"""
import argparse
import json
import sys
from typing import Dict, Any


def extract_features(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Map common Suricata keys to model features. This is best-effort; adapt as needed."""
    # Try several common key locations
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
        # try nested flow/tcp
        flow = alert.get('flow') if isinstance(alert.get('flow'), dict) else None
        if flow:
            flag = flow.get('tcp_flags')
    flag = flag or 'unknown'

    src_bytes = find('tx_bytes', 'src_bytes', default=0) or 0
    dst_bytes = find('rx_bytes', 'dst_bytes', default=0) or 0

    # Build sample dict
    sample = {
        'protocol_type': proto,
        'service': service,
        'flag': flag,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
    }
    return sample


def main():
    parser = argparse.ArgumentParser(description='Read Suricata alerts and predict using model or HTTP API')
    parser.add_argument('--input', '-i', help='Path to newline-delimited JSON alerts (use - for stdin)', required=True)
    parser.add_argument('--model', help='Path to local .joblib model (if provided, will call predict() directly)')
    parser.add_argument('--url', help='URL of /predict endpoint to POST to (if provided, will HTTP POST)')
    parser.add_argument('--batch', action='store_true', help='Send alerts in batches to the HTTP endpoint (only with --url)')
    args = parser.parse_args()

    if not args.model and not args.url:
        print('Error: provide --model to use local predict() or --url to POST to an API', file=sys.stderr)
        sys.exit(2)

    # Input stream
    if args.input == '-':
        stream = sys.stdin
    else:
        stream = open(args.input, 'r', encoding='utf-8')

    if args.url and args.batch:
        # Collect all alerts then POST as a list
        alerts = []
        for line in stream:
            if not line.strip():
                continue
            try:
                alert = json.loads(line)
            except Exception as e:
                print('skipping invalid json line:', e, file=sys.stderr)
                continue
            sample = extract_features(alert)
            alerts.append(sample)
        # Post to URL
        import requests
        resp = requests.post(args.url, json=alerts)
        print(resp.status_code)
        try:
            print(resp.json())
        except Exception:
            print(resp.text)
        return

    # Per-line processing
    if args.url:
        import requests
        for line in stream:
            if not line.strip():
                continue
            try:
                alert = json.loads(line)
            except Exception as e:
                print(json.dumps({'error': 'invalid_json', 'details': str(e)}))
                continue
            sample = extract_features(alert)
            try:
                resp = requests.post(args.url, json=sample, timeout=5)
                out = resp.json() if resp.headers.get('Content-Type','').startswith('application/json') else {'status': resp.status_code, 'text': resp.text}
            except Exception as e:
                out = {'error': 'request_failed', 'details': str(e)}
            print(json.dumps({'alert': alert, 'sample': sample, 'prediction': out}))
    else:
        # Local model predict() calls
        from app.models.infer import predict
        for line in stream:
            if not line.strip():
                continue
            try:
                alert = json.loads(line)
            except Exception as e:
                print(json.dumps({'error': 'invalid_json', 'details': str(e)}))
                continue
            sample = extract_features(alert)
            try:
                res = predict(args.model, sample)
            except Exception as e:
                res = {'error': 'predict_failed', 'details': str(e)}
            print(json.dumps({'alert': alert, 'sample': sample, 'prediction': res}))


if __name__ == '__main__':
    main()
