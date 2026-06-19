"""
LangChain tools wrapping the 5 MCP servers built in Phase 3.

Each tool makes a plain HTTP call to its corresponding FastAPI server.
This is the actual "MCP" integration point: the agent never imports
sensor_mcp.py / cmms_mcp.py etc. directly — it only ever speaks HTTP
to them, exactly as it would to a real external MCP server. That
separation is what makes it trivial to later point these same tools
at a real SAP PM instance or a real Twilio account: only the server
side changes, never the agent side.

Requires the 5 MCP servers to be running (see mcp_servers/run_all.py)
before the agent can use any of these tools.
"""

import os

import httpx
from langchain_core.tools import tool

SENSOR_URL     = os.getenv("SENSOR_MCP_URL",     "http://localhost:8001")
CMMS_URL       = os.getenv("CMMS_MCP_URL",       "http://localhost:8002")
INVENTORY_URL  = os.getenv("INVENTORY_MCP_URL",  "http://localhost:8003")
SCHEDULING_URL = os.getenv("SCHEDULING_MCP_URL", "http://localhost:8004")
NOTIFY_URL     = os.getenv("NOTIFY_MCP_URL",     "http://localhost:8005")

TIMEOUT = 10.0


def _get(url: str, params: dict | None = None) -> dict:
    r = httpx.get(url, params=params, timeout=TIMEOUT)
    if r.status_code >= 400:
        # Surface the MCP server's error detail back to the agent as a
        # string, rather than raising — the agent needs to SEE failures
        # (404, 409, etc.) as part of its reasoning, not crash on them.
        return {"error": True, "status_code": r.status_code, "detail": r.json().get("detail", r.text)}
    return r.json()


def _post(url: str, json_body: dict) -> dict:
    r = httpx.post(url, json=json_body, timeout=TIMEOUT)
    if r.status_code >= 400:
        return {"error": True, "status_code": r.status_code, "detail": r.json().get("detail", r.text)}
    return r.json()


def _patch(url: str, json_body: dict) -> dict:
    r = httpx.patch(url, json=json_body, timeout=TIMEOUT)
    if r.status_code >= 400:
        return {"error": True, "status_code": r.status_code, "detail": r.json().get("detail", r.text)}
    return r.json()


# ---------------------------------------------------------------------------
# sensor-mcp tools
# ---------------------------------------------------------------------------

@tool
def get_anomaly_score(machine_id: str) -> dict:
    """
    Get the BiLSTM anomaly score for a machine's most recent sensor window.

    Returns anomaly_score (0-1), severity ("normal"/"warning"/"critical"),
    sensor_flags (list of sensors reading abnormally vs. their own recent
    history), and latest_values for all 4 sensors.

    Always call this FIRST when assessing a machine — it's the primary
    signal for whether any further action is needed at all.
    """
    return _get(f"{SENSOR_URL}/anomaly_score/{machine_id}")


@tool
def get_sensor_readings(machine_id: str, last_n: int = 50) -> dict:
    """
    Get raw sensor readings (vibration_rms, temperature_c, pressure_bar,
    current_amp) for a machine's last `last_n` timesteps.

    Use this when you need to look at actual numbers — for example, to
    explain WHY a machine triggered a high anomaly score, or to decide
    which specific sensor's behavior matches a known fault pattern.
    """
    return _get(f"{SENSOR_URL}/readings/{machine_id}", params={"last_n": last_n})


@tool
def list_machines() -> list:
    """List all machine IDs currently being monitored."""
    return _get(f"{SENSOR_URL}/machines")


# ---------------------------------------------------------------------------
# cmms-mcp tools
# ---------------------------------------------------------------------------

@tool
def create_work_order(machine_id: str, fault_type: str, description: str, priority: str = "medium") -> dict:
    """
    Create a maintenance work order in the CMMS (maintenance management system).

    priority must be one of: "low", "medium", "high", "critical".

    Call this once you've decided a machine genuinely needs maintenance
    attention — this is the action that turns your assessment into a
    trackable maintenance record that a human technician will see.
    """
    return _post(f"{CMMS_URL}/work_orders", {
        "machine_id": machine_id,
        "fault_type": fault_type,
        "description": description,
        "priority": priority,
    })


@tool
def list_work_orders(machine_id: str | None = None, status: str | None = None) -> list:
    """
    List existing work orders, optionally filtered by machine_id and/or
    status ("open", "assigned", "in_progress", "completed").

    Use this BEFORE creating a new work order to check whether one
    already exists for this machine/fault — avoid creating duplicates.
    """
    params = {}
    if machine_id:
        params["machine_id"] = machine_id
    if status:
        params["status"] = status
    return _get(f"{CMMS_URL}/work_orders", params=params)


# ---------------------------------------------------------------------------
# inventory-mcp tools
# ---------------------------------------------------------------------------

@tool
def check_part_stock(part_name: str) -> dict:
    """
    Check current stock level for a spare part.

    Known part names: bearing_6205, bearing_seal_kit, compressor_valve,
    valve_gasket_set, motor_winding_kit, cooling_fan_motor,
    lubricant_grade_2, thermal_sensor.

    Always check stock BEFORE creating a work order that depends on a
    specific part, so you can flag a stock shortage as part of your
    assessment rather than assuming the part will be available.
    """
    return _get(f"{INVENTORY_URL}/parts/{part_name}")


@tool
def order_part(part_name: str, quantity: int, machine_id: str | None = None) -> dict:
    """
    Order a spare part — decrements stock and logs the order.

    Will fail with an error if the requested quantity exceeds available
    stock — check check_part_stock first and handle that case (e.g.
    order what's available and flag the shortfall, or escalate).
    """
    body = {"part_name": part_name, "quantity": quantity}
    if machine_id:
        body["machine_id"] = machine_id
    return _post(f"{INVENTORY_URL}/parts/order", body)


# ---------------------------------------------------------------------------
# scheduling-mcp tools
# ---------------------------------------------------------------------------

@tool
def find_available_technician(fault_type: str | None = None) -> list:
    """
    List available technicians, ranked with the best specialty match
    for the given fault_type first (falls back to general-purpose
    technicians if no specialist is free).

    Call this before dispatch_technician to choose who to assign.
    """
    params = {}
    if fault_type:
        params["fault_type"] = fault_type
    return _get(f"{SCHEDULING_URL}/technicians/available", params=params)


@tool
def dispatch_technician(technician_id: str, machine_id: str, work_order_id: int | None = None) -> dict:
    """
    Dispatch a technician to a machine. Marks them unavailable and logs
    the dispatch. This is typically your final action in a remediation
    sequence, after confirming the work order exists and parts are
    available (or on order).
    """
    body = {"technician_id": technician_id, "machine_id": machine_id}
    if work_order_id is not None:
        body["work_order_id"] = work_order_id
    return _post(f"{SCHEDULING_URL}/dispatch", body)


# ---------------------------------------------------------------------------
# notify-mcp tools
# ---------------------------------------------------------------------------

@tool
def send_alert(machine_id: str, message: str, priority: str = "medium", channel: str = "console") -> dict:
    """
    Send a human-readable alert about a machine.

    priority must be one of: "low", "medium", "high", "critical".
    channel must be one of: "console", "sms", "email", "slack" (mocked).

    Use this to notify a human supervisor of what you found and what
    actions you took — especially for "warning" or "critical" severity
    findings, even if you've already created a work order and dispatched
    a technician. Humans should always know what the agent did.
    """
    return _post(f"{NOTIFY_URL}/alert", {
        "machine_id": machine_id,
        "message": message,
        "priority": priority,
        "channel": channel,
    })


# ---------------------------------------------------------------------------
# Full tool list — what the agent is given
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    get_anomaly_score,
    get_sensor_readings,
    list_machines,
    create_work_order,
    list_work_orders,
    check_part_stock,
    order_part,
    find_available_technician,
    dispatch_technician,
    send_alert,
]
