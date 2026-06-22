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
| 3 | MCP servers (5× FastAPI) | ✅ |
| 4 | Agentic AI core (ReAct + Claude) | ✅ |
| 5 | Streamlit dashboard | ✅ |
| 6 | AWS deployment | ✅ |

---

## Phase 6 — AWS deployment

Three deployment targets, matching how the 5 MCP servers + the agent
naturally split by resource needs:

| Component | Deploys to | Why |
|---|---|---|
| sensor-mcp (real BiLSTM, needs torch) | Docker / ECS / EC2 | Too heavy for Lambda; SageMaker handles the model itself |
| BiLSTM model | SageMaker endpoint | Real-time inference, scales independently of the HTTP layer |
| cmms / inventory / scheduling / notify-mcp | Docker locally, or Lambda + Function URLs | Lightweight, stateless-ish, perfect Lambda fit |
| Agent orchestrator | Lambda, triggered by EventBridge | Runs the full agent on a schedule with zero human involvement |
| Dashboard | Docker locally, or Streamlit Community Cloud / EC2 | Needs to reach all 5 MCP servers + read their SQLite state |

### Local Docker (recommended first step — verifies everything before touching AWS)

```bash
docker compose up --build
```

This builds and runs all 5 MCP servers as containers — `sensor-mcp`
from `docker/Dockerfile.sensor` (includes the trained model + torch),
the other 4 sharing `docker/Dockerfile.lightweight` (parameterized by
the `MCP_SERVER_MODULE` env var per service). Each lightweight
service's SQLite state persists in a named Docker volume mounted at
`/app/data` — **not** at `/app/mcp_servers`, since mounting a volume
directly over the application code directory would hide the code
copied there at build time (this was a real bug caught and fixed
during development — see the design notes below).

### SageMaker (the model)

```bash
python aws/sagemaker/package_model.py   # builds aws/sagemaker/model.tar.gz
python aws/sagemaker/deploy.py --bucket your-bucket --role-arn arn:aws:iam::...:role/SageMakerExecutionRole
```

`aws/sagemaker/inference.py` implements SageMaker's four required entry
points (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) around the
exact same `BiLSTMAnomalyDetector` + `Normalizer` from Phase 2 — no
model logic is duplicated. Once deployed, `models/inference.py`'s new
`get_anomaly_score_remote()` calls the endpoint instead of local
weights, while `get_anomaly_score()` (used everywhere else in the
project) remains completely unchanged and local-only by default.

### Lambda (the 4 lightweight MCP servers)

```bash
pip install --platform manylinux2014_x86_64 --target aws/lambda/_build \
    --only-binary=:all: -r aws/lambda/requirements.txt
python aws/lambda/package.py            # builds aws/lambda/package.zip
python aws/lambda/deploy.py --role-arn arn:aws:iam::...:role/LambdaExecutionRole
```

`aws/lambda/handler.py` wraps each FastAPI app with
[Mangum](https://github.com/jordaneremieff/mangum), so the exact same
`mcp_servers/*.py` code runs unmodified under both uvicorn (local) and
Lambda. `sensor-mcp` deliberately stays off Lambda — torch + the model
weights make it a poor fit for Lambda's package size and cold-start
characteristics; it runs via Docker/ECS instead.

### EventBridge (the autonomous schedule)

```bash
python aws/eventbridge/deploy.py \
    --role-arn arn:aws:iam::...:role/LambdaExecutionRole \
    --trace-bucket your-trace-bucket \
    --anthropic-api-key sk-ant-...
```

Deploys an orchestrator Lambda (`aws/eventbridge/scheduler_handler.py`)
that EventBridge invokes every 15 minutes with zero human involvement —
it assesses every machine via the real agent and writes each trace to
S3. This is the project's "fully autonomous" deployment shape: once
live, machines get checked on a fixed schedule with no one clicking
"Run agent" in the dashboard.

### Dashboard

For a portfolio deployment, [Streamlit Community Cloud](https://streamlit.io/cloud)
is the simplest option (free, point it at the GitHub repo, set
`ANTHROPIC_API_KEY` and the 5 MCP server URLs as secrets). For a fully
self-contained AWS deployment, run it via Docker on EC2 behind nginx,
pointed at the Lambda Function URLs / SageMaker endpoint from above.

### Design decisions worth knowing for interviews

**Caught three real deployment bugs through actual testing, not just
code review.** A bare `*.db` pattern in `.dockerignore` would have
silently excluded `data/sensors.db` from the Docker build context —
verified with a script that replays `.dockerignore`'s glob patterns
against real file paths, not by eyeballing it. Neither Dockerfile
installed `httpx`, which every `docker-compose.yml` healthcheck
imports — every container would have reported unhealthy despite
working perfectly. And mounting a named volume directly at
`/app/mcp_servers` would have hidden the application code copied there
at build time — fixed by making each server's DB/log path configurable
via an environment variable (`CMMS_DB_PATH`, etc.), defaulting to the
exact original local-dev behavior when unset, verified against the
full Phase 1-5 test suite to confirm zero regressions.

**Verified Lambda packaging by actually extracting the zip and running
it** — not just inspecting file lists. `aws/lambda/handler.py`'s
`lifespan="off"` setting (correct for Lambda's stateless invocation
model) means FastAPI's lifespan context manager — which normally calls
each server's `init_db()` — never fires; the very first request would
have crashed with `no such table: work_orders`. Fixed by calling
`init_db()` explicitly once per cold start, then proved it by invoking
a real `POST /work_orders` through the Lambda handler and confirming
the table existed.

**The orchestrator Lambda only bundles what it actually imports** —
verified empirically by tracking `sys.modules` before and after the
real import chain (`data.store.get_all_machine_ids` +
`agent.react_agent.run_agent`), rather than assuming from reading
import statements. This caught that `models/` and `mcp_servers/` were
being bundled unnecessarily (the orchestrator talks to sensor-mcp over
HTTP, exactly like `agent/tools.py` always has), cutting the package in
half.

**SageMaker's inference code never duplicates model logic.**
`aws/sagemaker/inference.py`'s four entry points are a thin
SageMaker-shaped wrapper around the identical `BiLSTMAnomalyDetector`
and `Normalizer` classes from Phase 2 — the same model that's unit
tested in `tests/test_phase2.py` is what runs in the SageMaker
container, verified by actually packaging it, extracting the tarball,
and running inference against it exactly as the real container would.

---

## Phase 5 — Streamlit dashboard

A single-page dashboard, five tabs, that ties every previous phase
together into something a non-technical stakeholder (or an interviewer)
can actually click through.

```bash
# Recommended order:
python setup_phase1.py            # if not already done
python models/train.py            # if not already done
python mcp_servers/run_all.py &   # background, or separate terminal

streamlit run dashboard/app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

**First-run note:** Streamlit may ask an email/telemetry question on
its very first launch on a machine. If running it non-interactively
(e.g. piping output, CI, or certain remote shells) this prompt can
hang waiting for input — pre-empt it with:
```bash
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << 'EOF'
[server]
headless = true
[browser]
gatherUsageStats = false
EOF
```

### The five tabs

| Tab | Shows | Works without MCP servers? |
|---|---|---|
| 📊 Overview | Per-machine health cards: severity, anomaly score, latest sensor values | Partial — sensor values yes, anomaly score needs sensor-mcp |
| 📈 Sensor data | Time-series charts per sensor, with anomalous readings highlighted | Yes — reads `data/sensors.db` directly |
| 🧠 Agent trace | Step-by-step tool calls from saved agent runs — the auditability centerpiece | Yes — reads saved JSON traces |
| 📋 Action log | Work orders, parts/inventory, technician dispatches, alert history | Yes — reads each MCP server's SQLite DB directly |
| ▶ Run agent | Manually trigger a live agent run for any machine | No — needs all 5 MCP servers + `ANTHROPIC_API_KEY` |

### Files

| File | Purpose |
|---|---|
| `dashboard/app.py` | All UI/layout — five `render_*` functions, one per tab |
| `dashboard/data_access.py` | Every data read the dashboard needs, fully decoupled from layout |

### Design decisions worth knowing for interviews

**Degrades gracefully at every layer, never crashes.** Every function
in `data_access.py` returns an empty-but-correctly-columned DataFrame
(or a clear `{"error": ...}` dict) when its data source isn't available
yet — a fresh clone with no servers running and no data generated still
opens the dashboard cleanly, just showing "no data yet" messages instead
of stack traces. This was verified directly with Streamlit's `AppTest`
harness in both the empty-state and populated-state cases.

**Reads the MCP servers' SQLite files directly for display, but calls
the live HTTP API for actions.** `get_work_orders()`, `get_parts()`,
etc. read `cmms.db` / `inventory.db` / `scheduling.db` directly —
there's no reason to round-trip through HTTP just to display read-only
historical data. But the "Run agent" tab and the anomaly score on the
Overview tab go through the real HTTP API, because those represent
*actions* (or live model inference) the dashboard doesn't own.

**Caught a real bug via proper testing, not just manual clicking.**
`render_sensor_charts()` originally passed `fault_mask.any()` — a
`numpy.bool_` — directly into Plotly's `showlegend` property, which
newer Plotly versions reject outright (`numpy.bool_` isn't accepted
where a plain Python `bool` is expected). This only surfaces when the
function actually *executes* with real fault data present, which is
exactly what running the app through `streamlit.testing.v1.AppTest`
in the test suite catches and a simple `python -c "import app"` would
not. Fixed with an explicit `bool(...)` cast.

**The trace viewer is the differentiator.** Most "AI dashboard" demos
just show a final answer. The Agent trace tab renders every tool call
the agent made — input, output, and timestamp — in an expandable list.
This is what turns "the agent said it's fine" into "here's exactly
how the agent reached that conclusion, and a human can verify each step."

---

## Phase 4 — Agentic AI core

A LangChain agent powered by Claude that reasons over the 10 tools
exposed by the 5 MCP servers from Phase 3 — checking machine health,
deciding whether action is warranted, and if so, creating work orders,
checking parts stock, dispatching technicians, and sending alerts,
all autonomously.

```bash
# 1. Make sure ANTHROPIC_API_KEY is set in .env
cp .env.example .env   # then edit it

# 2. Start the 5 MCP servers (separate terminal, or background it)
python mcp_servers/run_all.py

# 3. Run the agent for a single machine
python agent/react_agent.py PUMP-01

# 4. Or run it across all machines and save traces to disk
python agent/run_all_machines.py

# Run tests (the 13 pure-logic tests run with no setup; the 10 tool
# tests additionally require the MCP servers from step 2 to be running)
pytest tests/test_phase4.py -v
```

### Files

| File | Purpose |
|---|---|
| `agent/tools.py` | 10 LangChain tools, each making a real HTTP call to one of the 5 MCP servers |
| `agent/prompts.py` | System prompt defining the agent's role and decision policy |
| `agent/react_agent.py` | Builds the Claude-powered agent and runs it end-to-end for one machine |
| `agent/trace.py` | Structured `AgentRunTrace` — the step-by-step reasoning record |
| `agent/run_all_machines.py` | CLI convenience script — runs the agent across every machine, saves JSON traces |

### How it works

```
HumanMessage("Assess PUMP-01")
        │
        ▼
   Claude reasons ──► calls get_anomaly_score("PUMP-01")
        │                       │
        │              sensor-mcp runs the real BiLSTM
        │                       │
        ▼                       ▼
   Claude reasons ──► decides severity = "warning"
        │
        ▼
   Claude reasons ──► calls get_sensor_readings, check_part_stock,
        │             find_available_technician, create_work_order,
        │             send_alert — in whatever order it judges sensible
        ▼
   Final summary (no more tool calls) ──► returned to the human
```

Every tool call and result along the way is captured in an
`AgentRunTrace` — this is the project's centerpiece: a human supervisor
can audit exactly *why* the agent made each decision, not just see the
final outcome.

### Design decisions worth knowing for interviews

**Tools talk HTTP, never Python imports.** `agent/tools.py` never
imports `sensor_mcp.py` or any other server module directly — every
tool is a plain `httpx` call to `http://localhost:800X`. This is what
makes the MCP pattern real rather than cosmetic: swapping a mock MCP
server for a real SAP PM or Twilio integration requires zero changes
on the agent side.

**Errors are data, not exceptions.** When a tool call 404s or 409s,
`tools.py` catches it and returns `{"error": True, "status_code": ...,
"detail": ...}` instead of raising. This matters because the agent
needs to *see* failures as part of its reasoning (e.g. "insufficient
stock, I'll flag this rather than dispatch") — an uncaught exception
would just crash the run with no useful signal.

**System prompt encodes a real decision policy, not just persona.**
The prompt in `prompts.py` doesn't just say "you are a maintenance
engineer" — it specifies exactly when to escalate (normal vs. warning
vs. critical), to check for existing work orders before creating
duplicates, and to always send a summary alert even when no action is
taken. This is what keeps the agent's behavior predictable and
auditable rather than improvised.

**Built on `create_agent` (LangChain 1.x), not the deprecated
`AgentExecutor`.** The agent loop (reason → tool call → reason → ...
→ final answer) is handled by LangChain's modern API, which returns a
clean list of messages that `react_agent.py` walks through to
reconstruct the full tool-call sequence for the trace.

**The trace, not just the final answer, is the deliverable.**
`AgentRunTrace` captures every `(tool_name, input, output)` triple in
order. This is exactly what Phase 5's dashboard will render — a human
supervisor sees the agent's full chain of reasoning, which is the
difference between "trust me" and genuine auditability.

---

## Phase 3 — MCP servers

Five small FastAPI services, each exposing one capability the agent can call as a tool — together they form the MCP layer the agent operates over.

```bash
# Run all 5 servers at once (recommended for local dev)
python mcp_servers/run_all.py

# Or run one at a time, in separate terminals
uvicorn mcp_servers.sensor_mcp:app     --port 8001 --reload
uvicorn mcp_servers.cmms_mcp:app       --port 8002 --reload
uvicorn mcp_servers.inventory_mcp:app  --port 8003 --reload
uvicorn mcp_servers.scheduling_mcp:app --port 8004 --reload
uvicorn mcp_servers.notify_mcp:app     --port 8005 --reload

# Run tests
pytest tests/test_phase3.py -v
```

**Note:** `sensor-mcp` imports PyTorch and loads the trained BiLSTM, so
it takes a few seconds longer to come up than the other four servers.
If you hit it immediately after starting `run_all.py`, give it ~5-8
seconds.

Once running, every server has interactive Swagger docs at
`http://localhost:<port>/docs` — useful for manually poking endpoints
before wiring the agent to them in Phase 4.

### The five servers

| Server | Port | Wraps / mocks | Key endpoints |
|---|---|---|---|
| `sensor-mcp` | 8001 | `reader.py` + `inference.py` (real BiLSTM) | `GET /anomaly_score/{machine_id}`, `GET /readings/{machine_id}` |
| `cmms-mcp` | 8002 | SAP PM / Maximo (mock) | `POST /work_orders`, `PATCH /work_orders/{id}/status` |
| `inventory-mcp` | 8003 | Spare parts stock system (mock) | `GET /parts/{part_name}`, `POST /parts/order` |
| `scheduling-mcp` | 8004 | Crew dispatch system (mock) | `GET /technicians/available`, `POST /dispatch` |
| `notify-mcp` | 8005 | Twilio/SendGrid/Slack (mock) | `POST /alert`, `GET /alerts` |

### Design decisions worth knowing for interviews

**`sensor-mcp` is the only server with real logic behind it** — the
other four are deliberately built as realistic mocks (SQLite-backed,
proper status codes, proper validation) so the *agent integration
pattern* can be fully demonstrated without needing live SAP/Maximo/
Twilio credentials. Swapping any mock for the real system only
requires rewriting that one server's internals — the agent-facing
HTTP contract doesn't change.

**Specialty-aware technician matching.** `scheduling-mcp`'s
`/technicians/available` endpoint ranks technicians whose specialty
exactly matches the fault type first, falling back to general-purpose
technicians — so the agent doesn't just dispatch "whoever's free."

**Stock validation before commitment.** `inventory-mcp` returns a 409
(not a silent failure) if the agent tries to order more of a part than
exists in stock — this is intentional: it forces the agent to reason
about an alternative (smaller order, different part, escalate) rather
than the tool quietly lying about what happened.

**Consistent error handling.** Every server returns proper HTTP status
codes (404 for missing resources, 409 for conflicts) with a `detail`
message — this matters because in Phase 4 the agent will see these
errors as tool outputs and needs to reason about them, not just see a
generic failure.

**Lifespan-based startup.** Each server seeds its own SQLite database
on startup using FastAPI's `lifespan` context manager (not the
deprecated `on_event("startup")`), so a fresh clone of the repo "just
works" the first time any server boots.

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
