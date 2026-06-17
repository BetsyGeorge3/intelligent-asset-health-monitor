# Intelligent Asset Health Monitor

Agentic AI system that monitors industrial equipment, reasons about failure risk,
and autonomously takes remediation actions via MCP tool servers.

**Stack:** Python · PyTorch · LangChain · Anthropic Claude · FastAPI · SQLite/InfluxDB · Streamlit · AWS

---

## Project structure

```
asset_health_monitor/
├── agent/              # Agentic AI core (Phase 4)
├── mcp_servers/        # FastAPI MCP tool servers (Phase 3)
├── models/             # BiLSTM anomaly detection model (Phase 2)
├── dashboard/          # Streamlit dashboard (Phase 5)
├── data/               # Data generation, store, reader
│   ├── generate_sensor_data.py
│   ├── store.py
│   └── reader.py
├── tests/              # Unit tests
├── setup_phase1.py     # Phase 1 bootstrap script
├── requirements.txt
└── .env.example
```

---

## Phase 1 — Quick start

```bash
# 1. Clone / enter project
cd asset_health_monitor

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill in env vars
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY

# 5. Run Phase 1 setup (generates data, seeds DB, smoke-tests reader)
python setup_phase1.py

# 6. Run tests
pytest tests/test_phase1.py -v
```

Expected output from `setup_phase1.py`:
```
──────────────────────────────────────────────────────────
  Step 1 — Initialise database
──────────────────────────────────────────────────────────
Database ready at data/sensors.db

──────────────────────────────────────────────────────────
  Step 2 — Generate synthetic sensor data (30 days, 5-min intervals)
──────────────────────────────────────────────────────────
  Rows generated : 25,920
  Anomaly rows   : 2,592  (10.0%)
  Machines       : ['PUMP-01', 'COMPRESSOR-01', 'MOTOR-01']
  Fault types    : ['bearing_wear', 'valve_leak', 'overheating']
...
```

---

## Sensors simulated

| Sensor | Unit | Description |
|---|---|---|
| `vibration_rms` | mm/s | RMS vibration from accelerometer |
| `temperature_c` | °C | Bearing/housing temperature |
| `pressure_bar` | bar | Discharge pressure |
| `current_amp` | A | Motor current draw |

## Machines simulated

| Machine | Type | Fault injected |
|---|---|---|
| PUMP-01 | Centrifugal pump | Bearing wear (day 5) |
| COMPRESSOR-01 | Reciprocating compressor | Valve leak (day 12) |
| MOTOR-01 | Induction motor | Overheating (day 20) |

---

## Phases

| Phase | What gets built | Status |
|---|---|---|
| 1 | Data foundation | ✅ |
| 2 | BiLSTM anomaly model | ✅ |
| 3 | MCP servers (5× FastAPI) | 🔜 |
| 4 | Agentic AI core (ReAct + Claude) | 🔜 |
| 5 | Streamlit dashboard | 🔜 |
| 6 | AWS deployment | 🔜 |

---

## Phase 2 — BiLSTM anomaly detector

```bash
# Train the model (generates data, builds windows, trains, evaluates, saves)
python models/train.py

# Optional flags
python models/train.py --epochs 30 --lr 0.0005 --hidden-size 128

# Run inference on the latest reading for every machine
python models/inference.py

# Run tests
pytest tests/test_phase2.py -v
```

### Architecture

A bidirectional LSTM binary classifier:

```
Input (50 timesteps × 4 sensors)
  → BiLSTM (2 layers, 64 hidden units, bidirectional)
  → Concatenate final forward + backward hidden states
  → FC(128 → 64) → ReLU → Dropout
  → FC(64 → 1) → Sigmoid
  → Anomaly score [0, 1]
```

Bidirectionality matters here because a developing fault is a *ramp* —
reading the window both forwards and backwards helps the model
distinguish "rising toward failure" from "recovering from a fault,"
which a one-directional LSTM tends to blur together.

### Files

| File | Purpose |
|---|---|
| `models/bilstm.py` | Model architecture |
| `models/dataset.py` | Windowing, stratified train/val/test split, normalisation |
| `models/train.py` | Full training pipeline, saves weights + metadata |
| `models/inference.py` | Clean `get_anomaly_score(machine_id)` — what the agent calls |
| `models/saved/` | Trained weights (`bilstm_weights.pt`) + metadata (`model_metadata.json`) |

### Key design decisions

**Weighted loss.** Anomalies are only ~5% of the data. Without
upweighting the positive class, the model could hit 95% accuracy by
predicting "normal" for everything — useless in practice. The loss
weights the anomaly class by `(1 - pos_rate) / pos_rate`.

**Stratified splitting.** Train/val/test splits preserve the same
anomaly ratio in each split, so the validation F1 score is a reliable
signal rather than noise from an unlucky split.

**Selected by F1, not accuracy.** Accuracy is misleading on imbalanced
data. The best checkpoint is the one with the highest validation F1
(precision + recall balance), and that checkpoint is what gets
evaluated on the held-out test set.

**Normaliser fit on train only.** Z-score statistics are computed
exclusively from the training split and saved alongside the model
weights — this avoids leaking validation/test statistics into training
and guarantees inference always uses the exact same scaling the model
was trained on.

### Benchmark results (30 days synthetic data, 20 epochs)

| Metric | Test set |
|---|---|
| Precision | 1.00 |
| Recall | 0.88 |
| F1 | 0.94 |
| Accuracy | 0.99 |

Zero false positives on the test set — the model never cried wolf —
while catching 88% of genuine fault windows.
