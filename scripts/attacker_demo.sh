#!/usr/bin/env bash
# Attacker demo script — run on the attacker VM against a controlled victim VM
# Usage: ./scripts/attacker_demo.sh <VICTIM_IP>

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <VICTIM_IP>" >&2
  exit 2
fi

VICTIM=$1

echo "Attacker demo starting against $VICTIM"

echo "1) Ping sweep (baseline)"
ping -c 3 "$VICTIM" || true
sleep 1

echo "2) Simple HTTP GET (benign)"
curl -s -m 5 "http://$VICTIM/" >/dev/null || true
sleep 1

echo "3) SYN scan (nmap, limited ports)"
if command -v nmap >/dev/null 2>&1; then
  nmap -sS -T4 -p 22,80,443 "$VICTIM" || true
else
  echo "nmap not installed — skipping" >&2
fi
sleep 2

echo "4) Controlled SYN packets (hping3, limited count)"
if command -v hping3 >/dev/null 2>&1; then
  sudo hping3 -S -p 80 -c 200 "$VICTIM" || true
else
  echo "hping3 not installed — skipping" >&2
fi
sleep 2

echo "5) Crafted FIN packets using scapy (small count)"
if python3 -c "import scapy" >/dev/null 2>&1; then
  python3 - <<'PY'
from scapy.all import IP, TCP, send
dst = "${VICTIM}"
for sport in range(4000,4005):
    pkt = IP(dst=dst)/TCP(sport=sport, dport=80, flags="F")
    send(pkt, count=3, verbose=False)
print('scapy packets sent')
PY
else
  echo "scapy not available — skipping" >&2
fi

echo "Attacker demo finished. Monitor the IDS dashboard for alerts."
