**IDS: SVM & Decision Tree — User Manual**

**Overview**

This repository implements a simple intrusion-detection pipeline using classical machine-learning models (SVM and Decision Tree) trained on the NSL-KDD dataset and a Suricata-based alert ingestion flow. It contains tools to:

- download and preprocess NSL-KDD,
- train and evaluate models (`app/models/`),
- convert Suricata alerts into model features, and
- replay/generate attacker traffic for a lab demo.

Use this manual to set up a **victim VM**, a **host VM** (generates attacks), run the project `main.py`, and observe how alerts are detected and classified by the ML models.

Important: Only test inside isolated lab networks or VMs you control and have permission to use.

**What you need**

- Two VMs (or two guest OS instances) on the same isolated virtual network: one victim, one attacker/host. Ubuntu 22.04 (or similar) is recommended.
- Python 3.8+ on any machine where the Python code runs.
- sudo access on the victim VM to install Suricata and capture packets.
- This repository copied into both VMs (or shared via a synced folder).

**Repository layout (short)**

- `main.py` — orchestrates the demo pipeline.
- `app/data/` — dataset utilities (download + make dataset).
- `app/features/` — feature columns and helpers.
- `app/models/` — training, evaluation, and saved model artifacts.
- `app/helpers/` — runtime helpers for Suricata ingestion and live prediction (`ids_suricata.py`).
- `data/` and `data/interim/` — downloaded and processed datasets.

1. Prepare VMs & network

---

Goal: victim and host must be able to exchange traffic on the test interface but remain isolated from production networks.

- In your VM manager (VirtualBox, VMware, KVM, cloud), create two VMs on the same internal/host-only network. Assign static IPs (example):
  - Victim: 192.168.50.10
  - Host/Attacker: 192.168.50.20

- Ensure the network interface you will capture on (e.g., `eth0`) is the one tied to the isolated network.

2. Install prerequisites (on both VMs where Python scripts run)

---

Clone the repo and create a Python virtual environment:

```bash
git clone https://github.com/mubex-dot/ids_project.git
cd ids_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the `main.py` file to train the models with the dataset

<!-- If you only want to run the demo without training, the pre-trained models in `models/` are sufficient. -->

3. Victim VM — Suricata install & configure

---

On the victim VM (requires sudo):

```bash
# Install Suricata and optional helpers
sudo apt update
sudo apt install -y suricata tcpreplay tshark

# (Optional) inspect `app/helpers/ids_suricata.py` for helper utilities used by the project

# Start suricata on the test interface (replace <iface>)
sudo suricata -c /etc/suricata/suricata.yaml -i <iface> -l /var/log/suricata

# Confirm logs are produced
sudo tail -f /var/log/suricata/eve.json
```

Notes:

- Suricata writes alerts to `/var/log/suricata/eve.json` by default (JSON). This repo can convert those logs using `app/helpers/ids_suricata.py` or by running the helper on the victim machine.

4. Host VM — generate/replay attacker traffic

---

On the host/attacker VM (must be able to reach the victim's IP):

Use standard traffic-generation tools; the repo's demo scripts are optional and may be deleted. Common options:

```bash
# Replay a PCAP to the victim IP using tcpreplay
sudo tcpreplay --intf1=<iface> sample_attack.pcap

# Or generate traffic via scapy (Python) or use hping3
python3 - <<'PY'
from scapy.all import *
pkt = IP(dst='192.168.50.10')/TCP(dport=80)/Raw(b'GET / HTTP/1.1\\r\\nHost: victim\\r\\n\\r\\n')
send(pkt, inter=0.01, count=200)
PY
```

Check the victim's Suricata logs (`eve.json`) while replaying traffic — you should see alerts appear.

5. Convert Suricata logs to model input

---

The repository provides `app/helpers/ids_suricata.py` which can read Suricata `eve.json`, extract NSL-KDD-like features, and either write an alerts file or run predictions directly using the saved models.

Example (batch processing or live tail):

```bash
source .venv/bin/activate
python3 app/helpers/ids_suricata.py --eve /var/log/suricata/eve.json --model models/best_svm.joblib --alert-file ids_alerts.jsonl
```

If the victim VM produced logs in `/var/log/suricata`, copy `eve.json` to the machine where the repo and models live (or run the helper on the victim VM if Python deps are installed there).

6. Run inference / classification

---

Two quick options to classify alerts:

Direct inference call (example):

```bash
python app/models/infer.py --logfile ids_alerts.jsonl --output results.json --verbose
```

The `reports/alerts_predicted.csv` (or similar) will include predicted classes/labels for each converted alert.

7. Run the entire pipeline locally (optional)

---

If you want to run download → preprocess → train → evaluate locally (training can take time):

```bash
source .venv/bin/activate
python3 main.py
```

`main.py` executes the dataset download, preprocessing, training (fast sampling by default), and evaluation. If network download fails, you can manually place `KDDTrain+.txt` and `KDDTest+.txt` in `data/raw/`.

8. Common issues & troubleshooting

---

- pandas `.str` AttributeError during preprocessing: ensure the `label` column is actually string typed. The project includes a fix that coerces label values to strings and handles malformed files.
- Training errors ("all fits failed" / "could not convert string to float"): ensure `X` passed to the preprocessor contains numeric columns only, and the `label` and `target` columns are excluded. The training code filters categorical and numeric columns; if your CSV contains misaligned columns, double-check `data/interim/*.csv` and column headers.
- Missing Python packages: run `pip install -r requirements.txt` inside the activated virtualenv.
- Suricata doesn't show alerts: confirm capture interface and that attack traffic reached the victim IP. Use `tcpdump -i <iface>` to observe traffic.

9. Extending and customizing

---

- Add new features in `app/features/` and update `EXPECTED_FEATURES`.
- Add additional detectors or persist alerts to a SIEM using `static/dashboard.html` or by integrating `reports/` outputs into your visualization stack.

10. Safety & legal reminder

---

Always perform testing in controlled lab environments and obtain written permission before testing on any network you do not own.

11. Helpful commands summary

---

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Local end-to-end demo
python3 main.py

# Start Suricata (victim)
sudo suricata -c /etc/suricata/suricata.yaml -i <iface> -l /var/log/suricata

# Replay a PCAP from attacker host
sudo tcpreplay --intf1=<iface> sample_attack.pcap

# Use the Suricata helper for live processing + prediction
python3 app/helpers/ids_suricata.py --eve /var/log/suricata/eve.json --model models/best_svm.joblib

# Run inference on a prepared CSV/JSON
python app/models/infer.py --logfile ids_alerts.jsonl --output results.json --verbose
```

## Files of interest

- [main.py](main.py) — orchestration entrypoint
- [app/helpers/ids_suricata.py](app/helpers/ids_suricata.py) — Suricata ingestion and live prediction helper
- [app/models/infer.py](app/models/infer.py) — inference runtime
- [app/data/make_dataset.py](app/data/make_dataset.py) — dataset preprocessing (label handling)

## Contact / contribution

Open an issue or PR if you want help adapting the repo to your environment or adding features. Include OS, VM provider, and the steps you followed.

## Acknowledgements

This project is intended for defensive research and lab-based education only.
