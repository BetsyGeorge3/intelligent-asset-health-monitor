"""
Sensor data store — SQLite backend.

Provides two functions the rest of the project uses:
    write_readings(df)          — bulk-load a DataFrame
    get_readings(machine_id, n) — fetch the last N readings for a machine

SQLite is used here for simplicity. Swap the backend for InfluxDB in production
by replacing these two functions — the interface stays identical.
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("SQLITE_PATH", "data/sensors.db"))

SENSORS = ["vibration_rms", "temperature_c", "pressure_bar", "current_amp"]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sensor_readings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    machine_id   TEXT    NOT NULL,
    machine_type TEXT    NOT NULL,
    vibration_rms REAL,
    temperature_c REAL,
    pressure_bar  REAL,
    current_amp   REAL,
    fault_type   TEXT
);
CREATE INDEX IF NOT EXISTS idx_machine_ts
    ON sensor_readings (machine_id, timestamp DESC);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with _conn() as con:
        con.executescript(CREATE_TABLE)
    print(f"Database ready at {DB_PATH}")


def write_readings(df: pd.DataFrame) -> int:
    """
    Insert a DataFrame of sensor readings into the store.

    Args:
        df: DataFrame produced by generate_sensor_data.generate()

    Returns:
        Number of rows inserted.
    """
    cols = ["timestamp", "machine_id", "machine_type"] + SENSORS + ["fault_type"]
    df   = df[cols].copy()
    df["timestamp"] = df["timestamp"].astype(str)

    with _conn() as con:
        df.to_sql("sensor_readings", con, if_exists="append", index=False)

    return len(df)


def get_readings(machine_id: str, last_n: int = 50) -> pd.DataFrame:
    """
    Fetch the most recent `last_n` readings for a machine.

    Args:
        machine_id: e.g. "PUMP-01"
        last_n: number of rows to return (newest first)

    Returns:
        DataFrame sorted ascending by timestamp (oldest first).
    """
    query = """
        SELECT timestamp, machine_id, machine_type,
               vibration_rms, temperature_c, pressure_bar, current_amp,
               fault_type
        FROM   sensor_readings
        WHERE  machine_id = ?
        ORDER  BY timestamp DESC
        LIMIT  ?
    """
    with _conn() as con:
        df = pd.read_sql_query(query, con, params=(machine_id, last_n))

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_all_machine_ids() -> list[str]:
    """Return list of distinct machine IDs in the store."""
    with _conn() as con:
        rows = con.execute(
            "SELECT DISTINCT machine_id FROM sensor_readings ORDER BY machine_id"
        ).fetchall()
    return [r[0] for r in rows]


def get_latest_reading(machine_id: str) -> dict | None:
    """Return the single most recent reading for a machine as a dict."""
    df = get_readings(machine_id, last_n=1)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


# --- CLI helper ----------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.generate_sensor_data import generate

    print("Initialising database...")
    init_db()

    print("Generating 30 days of synthetic data...")
    df = generate(days=30)

    print(f"Writing {len(df):,} rows to {DB_PATH}...")
    n = write_readings(df)
    print(f"Done — {n:,} rows written.")

    print("\nSpot-check: last 3 readings for PUMP-01")
    sample = get_readings("PUMP-01", last_n=3)
    print(sample.to_string(index=False))
