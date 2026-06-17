"""
inventory-mcp — mock spare parts inventory system.

Before the agent dispatches a technician, it needs to know whether the
required part is actually in stock. This server seeds a small parts
catalog relevant to the three machines (pump, compressor, motor) and
lets the agent check stock and place orders.

Endpoints:
    GET  /parts                 → list all parts and stock levels
    GET  /parts/{part_name}     → check stock for one part
    POST /parts/order           → decrement stock, log an order

Run:
    uvicorn mcp_servers.inventory_mcp:app --port 8003 --reload
"""

import sqlite3
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp_servers.schemas import now_iso

DB_PATH = Path(__file__).parent / "inventory.db"

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS parts (
    part_name    TEXT PRIMARY KEY,
    description  TEXT NOT NULL,
    quantity     INTEGER NOT NULL,
    reorder_level INTEGER NOT NULL,
    unit_cost_aed REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    part_name   TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    machine_id  TEXT,
    ordered_at  TEXT NOT NULL
);
"""

# Seed data — parts relevant to PUMP-01, COMPRESSOR-01, MOTOR-01
SEED_PARTS = [
    ("bearing_6205",        "Deep groove ball bearing 6205",       12, 5,  85.0),
    ("bearing_seal_kit",    "Bearing housing seal kit",             8, 3,  45.0),
    ("compressor_valve",    "Reciprocating compressor discharge valve", 4, 2, 620.0),
    ("valve_gasket_set",    "Valve gasket set",                    15, 5,  30.0),
    ("motor_winding_kit",   "Induction motor rewind kit",           2, 1, 1450.0),
    ("cooling_fan_motor",   "Motor cooling fan assembly",           6, 2,  210.0),
    ("lubricant_grade_2",   "Industrial bearing grease, grade 2",  20, 8,  18.0),
    ("thermal_sensor",      "RTD temperature sensor probe",        10, 4,  65.0),
]


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.executescript(CREATE_TABLES)
        existing = con.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
        if existing == 0:
            con.executemany(
                "INSERT INTO parts (part_name, description, quantity, reorder_level, unit_cost_aed) "
                "VALUES (?, ?, ?, ?, ?)",
                SEED_PARTS,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="inventory-mcp", version="1.0.0", lifespan=lifespan)


class Part(BaseModel):
    part_name: str
    description: str
    quantity: int
    reorder_level: int
    unit_cost_aed: float
    low_stock: bool


class OrderRequest(BaseModel):
    part_name: str
    quantity: int = Field(gt=0)
    machine_id: str | None = None


class OrderResponse(BaseModel):
    id: int
    part_name: str
    quantity_ordered: int
    remaining_stock: int
    machine_id: str | None
    ordered_at: str


def _row_to_part(row: sqlite3.Row) -> Part:
    d = dict(row)
    d["low_stock"] = d["quantity"] <= d["reorder_level"]
    return Part(**d)


@app.get("/parts", response_model=list[Part])
def list_parts():
    """List all parts with current stock levels."""
    with _conn() as con:
        rows = con.execute("SELECT * FROM parts ORDER BY part_name").fetchall()
    return [_row_to_part(r) for r in rows]


@app.get("/parts/{part_name}", response_model=Part)
def get_part(part_name: str):
    """Check stock level for a specific part."""
    with _conn() as con:
        row = con.execute("SELECT * FROM parts WHERE part_name = ?", (part_name,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Part '{part_name}' not found in catalog")
    return _row_to_part(row)


@app.post("/parts/order", response_model=OrderResponse)
def order_part(order: OrderRequest):
    """
    Place an order for a part — decrements stock and logs the order.
    The agent calls this after confirming a part is needed and in stock.
    """
    with _conn() as con:
        row = con.execute("SELECT * FROM parts WHERE part_name = ?", (order.part_name,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Part '{order.part_name}' not found in catalog")

        current_qty = row["quantity"]
        if current_qty < order.quantity:
            raise HTTPException(
                status_code=409,
                detail=f"Insufficient stock for '{order.part_name}': "
                       f"requested {order.quantity}, available {current_qty}",
            )

        new_qty = current_qty - order.quantity
        ts = now_iso()
        con.execute("UPDATE parts SET quantity = ? WHERE part_name = ?", (new_qty, order.part_name))
        cur = con.execute(
            "INSERT INTO orders (part_name, quantity, machine_id, ordered_at) VALUES (?, ?, ?, ?)",
            (order.part_name, order.quantity, order.machine_id, ts),
        )
        order_id = cur.lastrowid

    return OrderResponse(
        id=order_id,
        part_name=order.part_name,
        quantity_ordered=order.quantity,
        remaining_stock=new_qty,
        machine_id=order.machine_id,
        ordered_at=ts,
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "inventory-mcp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
