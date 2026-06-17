"""
Phase 1 setup script.

Run this once to:
  1. Initialise the SQLite database
  2. Generate 30 days of synthetic sensor data
  3. Load it into the database
  4. Smoke-test the reader utility

Usage:
    cd asset_health_monitor
    python setup_phase1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.generate_sensor_data import generate
from data.store import init_db, write_readings, get_readings, get_all_machine_ids
from data.reader import load_window, sensor_context_string


def separator(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def main():
    separator("Step 1 — Initialise database")
    init_db()

    separator("Step 2 — Generate synthetic sensor data (30 days, 5-min intervals)")
    df = generate(days=30, freq_minutes=5)
    total     = len(df)
    anomalies = df[df["fault_type"].notna()]
    print(f"  Rows generated : {total:,}")
    print(f"  Anomaly rows   : {len(anomalies):,}  ({len(anomalies)/total*100:.1f}%)")
    print(f"  Machines       : {df['machine_id'].unique().tolist()}")
    print(f"  Fault types    : {anomalies['fault_type'].dropna().unique().tolist()}")

    separator("Step 3 — Write to database")
    n = write_readings(df)
    print(f"  {n:,} rows written successfully.")

    separator("Step 4 — Smoke-test reader utility")
    machine_ids = get_all_machine_ids()
    print(f"  Machines in DB : {machine_ids}")

    for mid in machine_ids:
        print()
        window = load_window(mid, last_n=50)
        print(sensor_context_string(window))
        print(f"  Numpy array shape : {window.as_array.shape}  (samples × sensors)")

    separator("Phase 1 complete!")
    print("  Next step → Phase 2: build the BiLSTM anomaly model.")
    print("  Run: python models/train.py\n")


if __name__ == "__main__":
    main()
