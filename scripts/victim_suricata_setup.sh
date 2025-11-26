#!/usr/bin/env bash
# Victim Suricata quick setup helper (Debian/Ubuntu)
# Run as root or with sudo on the victim VM to install and enable JSON logging

set -euo pipefail

echo "This script installs Suricata (Debian/Ubuntu) and ensures JSON eve logging is enabled."
echo "Review /etc/suricata/suricata.yaml after install and tailor it to your environment."

if [ "$(id -u)" -ne 0 ]; then
  echo "Please run as root or with sudo: sudo $0" >&2
  exit 1
fi

if ! command -v suricata >/dev/null 2>&1; then
  apt-get update
  apt-get install -y suricata
fi

# Enable EVE JSON output (this is a conservative edit — inspect after running)
YAML=/etc/suricata/suricata.yaml
if [ -f "$YAML" ]; then
  echo "Configuring $YAML to ensure eve-json is enabled (a minimal check)."
  # This does not attempt to fully manage YAML structure — open the file for review.
  grep -n "eve-log" "$YAML" || true
  echo "Ensure the 'eve-log' output is enabled and set to type: json in $YAML"
fi

echo "Starting Suricata (service)"
systemctl enable --now suricata || true

echo "Suricata should be running. Check its JSON log at /var/log/suricata/eve.json"
echo "If your IDS host monitors a different path, set: IDS_SURICATA_LOG=/var/log/suricata/eve.json"
