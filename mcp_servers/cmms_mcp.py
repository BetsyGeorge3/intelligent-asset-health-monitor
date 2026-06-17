"""
cmms-mcp — mock Computerized Maintenance Management System
(stands in for SAP PM, IBM Maximo, or similar).

Lets the agent create and query work orders. This is what gives the
agent's decisions real-world consequence: when it decides a machine
needs attention, this is the tool that turns that decision into a
trackable maintenance record.

Storage: a small SQLite table, separate from the sensor database —
in a real deployment this MCP server would instead call out to the
actual CMMS's REST API, but the FastAPI interface the agent sees
would look identical.

Endpoints:
    POST /work_orders             → create a new work order
    GET  /work_orders              → list all work orders (optional filters)
    GET  /work_orders/{id}         → fetch one work order
    PATCH /work_orders/{id}/status → update status

Run:
    uvicorn mcp_servers.cmms_mcp:app --port 8002 --reload
"""

import sqlite3
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_servers.schemas import Priority, WorkOrderStatus, now_iso

DB_PATH = Path(__file__).parent / "cmms.db"


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS work_orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id   TEXT    NOT NULL,
    fault_type   TEXT    NOT NULL,
    description  TEXT    NOT NULL,
    priority     TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'open',
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.execute(CREATE_TABLE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="cmms-mcp", version="1.0.0", lifespan=lifespan)


class WorkOrderCreate(BaseModel):
    machine_id:  str
    fault_type:  str
    description: str
    priority:    Priority = Priority.medium


class WorkOrder(BaseModel):
    id: int
    machine_id: str
    fault_type: str
    description: str
    priority: str
    status: str
    created_at: str
    updated_at: str


class StatusUpdate(BaseModel):
    status: WorkOrderStatus


def _row_to_workorder(row: sqlite3.Row) -> WorkOrder:
    return WorkOrder(**dict(row))


@app.post("/work_orders", response_model=WorkOrder)
def create_work_order(wo: WorkOrderCreate):
    """
    Create a new work order. The agent calls this once it has decided
    a machine needs maintenance action.
    """
    ts = now_iso()
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO work_orders
               (machine_id, fault_type, description, priority, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'open', ?, ?)""",
            (wo.machine_id, wo.fault_type, wo.description, wo.priority.value, ts, ts),
        )
        new_id = cur.lastrowid
        row = con.execute("SELECT * FROM work_orders WHERE id = ?", (new_id,)).fetchone()

    return _row_to_workorder(row)


@app.get("/work_orders", response_model=list[WorkOrder])
def list_work_orders(machine_id: str | None = None, status: str | None = None):
    """List work orders, optionally filtered by machine_id and/or status."""
    query = "SELECT * FROM work_orders WHERE 1=1"
    params = []
    if machine_id:
        query += " AND machine_id = ?"
        params.append(machine_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC"

    with _conn() as con:
        rows = con.execute(query, params).fetchall()
    return [_row_to_workorder(r) for r in rows]


@app.get("/work_orders/{work_order_id}", response_model=WorkOrder)
def get_work_order(work_order_id: int):
    with _conn() as con:
        row = con.execute("SELECT * FROM work_orders WHERE id = ?", (work_order_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Work order {work_order_id} not found")
    return _row_to_workorder(row)


@app.patch("/work_orders/{work_order_id}/status", response_model=WorkOrder)
def update_status(work_order_id: int, update: StatusUpdate):
    ts = now_iso()
    with _conn() as con:
        existing = con.execute("SELECT * FROM work_orders WHERE id = ?", (work_order_id,)).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Work order {work_order_id} not found")
        con.execute(
            "UPDATE work_orders SET status = ?, updated_at = ? WHERE id = ?",
            (update.status.value, ts, work_order_id),
        )
        row = con.execute("SELECT * FROM work_orders WHERE id = ?", (work_order_id,)).fetchone()
    return _row_to_workorder(row)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "cmms-mcp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
