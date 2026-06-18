"""
scheduling-mcp — mock crew scheduling / dispatch system.

Lets the agent check which technicians are available and dispatch one
to handle a work order. Technician specialties are matched against
fault types so the agent picks someone qualified, not just anyone free.

Endpoints:
    GET  /technicians                       → list all technicians
    GET  /technicians/available             → filter by specialty + availability
    POST /dispatch                          → assign a technician to a work order

Run:
    uvicorn mcp_servers.scheduling_mcp:app --port 8004 --reload
"""

import sqlite3
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_servers.schemas import now_iso

DB_PATH = Path(__file__).parent / "scheduling.db"

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS technicians (
    technician_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    specialty     TEXT NOT NULL,
    available     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS dispatches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    technician_id TEXT NOT NULL,
    work_order_id INTEGER,
    machine_id    TEXT NOT NULL,
    dispatched_at TEXT NOT NULL
);
"""

# Seed crew — specialties map to the fault types in our 3 machines
SEED_TECHNICIANS = [
    ("TECH-01", "Ahmed Al Mansoori", "bearing_wear",  1),
    ("TECH-02", "Fatima Al Shamsi",  "valve_leak",    1),
    ("TECH-03", "Rashid Al Nuaimi",  "overheating",   1),
    ("TECH-04", "Sara Al Qassimi",   "general",       1),
]


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.executescript(CREATE_TABLES)
        existing = con.execute("SELECT COUNT(*) FROM technicians").fetchone()[0]
        if existing == 0:
            con.executemany(
                "INSERT INTO technicians (technician_id, name, specialty, available) VALUES (?, ?, ?, ?)",
                SEED_TECHNICIANS,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="scheduling-mcp", version="1.0.0", lifespan=lifespan)


class Technician(BaseModel):
    technician_id: str
    name: str
    specialty: str
    available: bool


class DispatchRequest(BaseModel):
    technician_id: str
    machine_id: str
    work_order_id: int | None = None


class DispatchResponse(BaseModel):
    id: int
    technician_id: str
    technician_name: str
    machine_id: str
    work_order_id: int | None
    dispatched_at: str


def _row_to_technician(row: sqlite3.Row) -> Technician:
    d = dict(row)
    d["available"] = bool(d["available"])
    return Technician(**d)


@app.get("/technicians", response_model=list[Technician])
def list_technicians():
    """List all technicians and their current availability."""
    with _conn() as con:
        rows = con.execute("SELECT * FROM technicians ORDER BY technician_id").fetchall()
    return [_row_to_technician(r) for r in rows]


@app.get("/technicians/available", response_model=list[Technician])
def available_technicians(fault_type: str | None = None):
    """
    Return available technicians, optionally filtered by specialty
    matching the given fault_type. Falls back to 'general' specialty
    technicians if no exact specialty match exists.
    """
    with _conn() as con:
        if fault_type:
            rows = con.execute(
                "SELECT * FROM technicians WHERE available = 1 "
                "AND (specialty = ? OR specialty = 'general') "
                "ORDER BY (specialty != ?) ASC",  # exact specialty match first
                (fault_type, fault_type),
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM technicians WHERE available = 1").fetchall()
    return [_row_to_technician(r) for r in rows]


@app.post("/dispatch", response_model=DispatchResponse)
def dispatch_technician(req: DispatchRequest):
    """
    Assign a technician to a machine/work order. Marks them unavailable
    and logs the dispatch. The agent calls this as its final action once
    it has confirmed a part is in stock and a qualified technician exists.
    """
    with _conn() as con:
        tech = con.execute(
            "SELECT * FROM technicians WHERE technician_id = ?", (req.technician_id,)
        ).fetchone()
        if tech is None:
            raise HTTPException(status_code=404, detail=f"Technician '{req.technician_id}' not found")
        if not tech["available"]:
            raise HTTPException(status_code=409, detail=f"Technician '{req.technician_id}' is not available")

        ts = now_iso()
        con.execute("UPDATE technicians SET available = 0 WHERE technician_id = ?", (req.technician_id,))
        cur = con.execute(
            "INSERT INTO dispatches (technician_id, work_order_id, machine_id, dispatched_at) "
            "VALUES (?, ?, ?, ?)",
            (req.technician_id, req.work_order_id, req.machine_id, ts),
        )
        dispatch_id = cur.lastrowid

    return DispatchResponse(
        id=dispatch_id,
        technician_id=req.technician_id,
        technician_name=tech["name"],
        machine_id=req.machine_id,
        work_order_id=req.work_order_id,
        dispatched_at=ts,
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "scheduling-mcp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
