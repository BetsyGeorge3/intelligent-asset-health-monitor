"""
Dashboard data access layer.

Centralises every data source the dashboard needs to read from:
    - data/sensors.db          (via data.reader / data.store — same code Phase 1-2 use)
    - mcp_servers' SQLite DBs  (cmms.db, inventory.db, scheduling.db) — read directly
      for speed, since the dashboard just displays state, it doesn't need to go
      through HTTP for read-only views
    - mcp_servers/alerts.log   (notify-mcp's alert history)
    - agent/traces/*.json      (saved agent runs from agent/run_all_machines.py)

The dashboard ALSO calls the live MCP servers + agent directly for the
"Run agent now" button, since that's a real action, not just a read.

Keeping all of this in one module means app.py stays focused on layout.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.store import get_all_machine_ids, get_readings
from data.reader import load_window, sensor_context_string

CMMS_DB       = ROOT / "mcp_servers" / "cmms.db"
INVENTORY_DB  = ROOT / "mcp_servers" / "inventory.db"
SCHEDULING_DB = ROOT / "mcp_servers" / "scheduling.db"
ALERTS_LOG    = ROOT / "mcp_servers" / "alerts.log"
TRACES_DIR    = ROOT / "agent" / "traces"

SENSOR_LABELS = {
    "vibration_rms":  "Vibration (mm/s RMS)",
    "temperature_c":  "Temperature (°C)",
    "pressure_bar":   "Pressure (bar)",
    "current_amp":    "Current (A)",
}


# ---------------------------------------------------------------------------
# Machines + sensors
# ---------------------------------------------------------------------------

def list_machines() -> list[str]:
    """All machine IDs known to the system. Empty list if DB not seeded yet."""
    try:
        return get_all_machine_ids()
    except Exception:
        return []


def get_machine_window(machine_id: str, last_n: int = 100) -> pd.DataFrame:
    """Raw sensor readings for charting — returns a tidy long-format DataFrame."""
    df = get_readings(machine_id, last_n=last_n)
    return df


def get_latest_snapshot(machine_id: str) -> dict | None:
    """Latest single reading for a machine — used by the overview cards."""
    df = get_readings(machine_id, last_n=1)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


# ---------------------------------------------------------------------------
# Anomaly scoring (calls sensor-mcp's live model — same as the agent does)
# ---------------------------------------------------------------------------

def get_anomaly_score_safe(machine_id: str, sensor_url: str = "http://localhost:8001") -> dict:
    """
    Fetch the BiLSTM anomaly score via the live sensor-mcp server.
    Falls back to a clear error dict (not an exception) if the server
    isn't running — the dashboard should degrade gracefully, not crash.
    """
    import httpx
    try:
        r = httpx.get(f"{sensor_url}/anomaly_score/{machine_id}", timeout=5.0)
        if r.status_code == 200:
            return r.json()
        return {"error": True, "detail": r.json().get("detail", r.text)}
    except httpx.ConnectError:
        return {"error": True, "detail": "sensor-mcp not reachable on port 8001 — is it running?"}
    except Exception as e:
        return {"error": True, "detail": str(e)}


# ---------------------------------------------------------------------------
# CMMS — work orders
# ---------------------------------------------------------------------------

def get_work_orders() -> pd.DataFrame:
    """All work orders, newest first. Empty DataFrame if cmms.db doesn't exist yet."""
    if not CMMS_DB.exists():
        return pd.DataFrame(columns=["id", "machine_id", "fault_type", "description",
                                      "priority", "status", "created_at", "updated_at"])
    con = sqlite3.connect(CMMS_DB)
    df = pd.read_sql_query("SELECT * FROM work_orders ORDER BY created_at DESC", con)
    con.close()
    return df


# ---------------------------------------------------------------------------
# Inventory — spare parts
# ---------------------------------------------------------------------------

def get_parts() -> pd.DataFrame:
    """Current parts catalog + stock levels."""
    if not INVENTORY_DB.exists():
        return pd.DataFrame(columns=["part_name", "description", "quantity",
                                      "reorder_level", "unit_cost_aed"])
    con = sqlite3.connect(INVENTORY_DB)
    df = pd.read_sql_query("SELECT * FROM parts ORDER BY part_name", con)
    con.close()
    df["low_stock"] = df["quantity"] <= df["reorder_level"]
    return df


def get_part_orders() -> pd.DataFrame:
    """Order history for parts."""
    if not INVENTORY_DB.exists():
        return pd.DataFrame(columns=["id", "part_name", "quantity", "machine_id", "ordered_at"])
    con = sqlite3.connect(INVENTORY_DB)
    df = pd.read_sql_query("SELECT * FROM orders ORDER BY ordered_at DESC", con)
    con.close()
    return df


# ---------------------------------------------------------------------------
# Scheduling — technicians + dispatches
# ---------------------------------------------------------------------------

def get_technicians() -> pd.DataFrame:
    if not SCHEDULING_DB.exists():
        return pd.DataFrame(columns=["technician_id", "name", "specialty", "available"])
    con = sqlite3.connect(SCHEDULING_DB)
    df = pd.read_sql_query("SELECT * FROM technicians ORDER BY technician_id", con)
    con.close()
    df["available"] = df["available"].astype(bool)
    return df


def get_dispatches() -> pd.DataFrame:
    if not SCHEDULING_DB.exists():
        return pd.DataFrame(columns=["id", "technician_id", "work_order_id", "machine_id", "dispatched_at"])
    con = sqlite3.connect(SCHEDULING_DB)
    df = pd.read_sql_query("SELECT * FROM dispatches ORDER BY dispatched_at DESC", con)
    con.close()
    return df


# ---------------------------------------------------------------------------
# Notify — alert history
# ---------------------------------------------------------------------------

def get_alerts() -> pd.DataFrame:
    """Alert log, newest first."""
    cols = ["machine_id", "message", "priority", "channel", "sent_at", "delivered"]
    if not ALERTS_LOG.exists():
        return pd.DataFrame(columns=cols)
    rows = []
    with open(ALERTS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    return df.sort_values("sent_at", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Agent traces — saved runs from agent/run_all_machines.py, or live runs
# triggered from the dashboard itself
# ---------------------------------------------------------------------------

def list_saved_traces() -> dict[str, dict]:
    """
    Load every saved trace JSON in agent/traces/.
    Returns { machine_id: trace_dict }.
    """
    traces = {}
    if not TRACES_DIR.exists():
        return traces
    for path in TRACES_DIR.glob("*.json"):
        try:
            with open(path) as f:
                trace = json.load(f)
            traces[trace["machine_id"]] = trace
        except Exception:
            continue
    return traces


def save_trace(trace_dict: dict) -> Path:
    """Persist a trace dict to agent/traces/{machine_id}.json — used after a live run."""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRACES_DIR / f"{trace_dict['machine_id']}.json"
    with open(out_path, "w") as f:
        json.dump(trace_dict, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# System health — are the MCP servers up? (for a small status indicator)
# ---------------------------------------------------------------------------

def check_mcp_servers() -> dict[str, bool]:
    """Ping all 5 MCP servers' /health endpoints. Returns {service_name: is_up}."""
    import httpx
    servers = {
        "sensor-mcp":     "http://localhost:8001/health",
        "cmms-mcp":       "http://localhost:8002/health",
        "inventory-mcp":  "http://localhost:8003/health",
        "scheduling-mcp": "http://localhost:8004/health",
        "notify-mcp":     "http://localhost:8005/health",
    }
    status = {}
    for name, url in servers.items():
        try:
            r = httpx.get(url, timeout=1.5)
            status[name] = r.status_code == 200
        except Exception:
            status[name] = False
    return status
