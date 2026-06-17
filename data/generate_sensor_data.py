"""
Synthetic sensor data generator.

Simulates 3 industrial machines with realistic sensor readings.
Injects faults so the anomaly model has something to learn.

Usage:
    python data/generate_sensor_data.py
    python data/generate_sensor_data.py --days 30 --output data/sensors.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- Machine definitions -------------------------------------------------------

MACHINES = {
    "PUMP-01": {
        "type": "centrifugal_pump",
        "normal": {
            "vibration_rms":  {"mean": 2.1,  "std": 0.15},  # mm/s
            "temperature_c":  {"mean": 65.0, "std": 1.5},   # °C
            "pressure_bar":   {"mean": 8.5,  "std": 0.2},   # bar
            "current_amp":    {"mean": 12.3, "std": 0.4},   # A
        },
    },
    "COMPRESSOR-01": {
        "type": "reciprocating_compressor",
        "normal": {
            "vibration_rms":  {"mean": 4.5,  "std": 0.3},
            "temperature_c":  {"mean": 85.0, "std": 2.0},
            "pressure_bar":   {"mean": 15.2, "std": 0.5},
            "current_amp":    {"mean": 28.7, "std": 0.8},
        },
    },
    "MOTOR-01": {
        "type": "induction_motor",
        "normal": {
            "vibration_rms":  {"mean": 1.8,  "std": 0.1},
            "temperature_c":  {"mean": 72.0, "std": 1.2},
            "pressure_bar":   {"mean": 0.0,  "std": 0.05},  # not applicable, near-zero
            "current_amp":    {"mean": 18.5, "std": 0.6},
        },
    },
}

# Fault scenarios — each has a start offset (days from start), duration (hours),
# and per-sensor multipliers that ramp up over the fault window.
FAULT_SCENARIOS = [
    {
        "machine_id": "PUMP-01",
        "fault_type": "bearing_wear",
        "start_day": 5,
        "duration_hours": 48,
        "ramp": {
            "vibration_rms": 3.5,   # rises to 3.5× normal
            "temperature_c": 1.25,  # rises 25%
            "pressure_bar":  0.92,  # slight pressure drop
            "current_amp":   1.10,
        },
    },
    {
        "machine_id": "COMPRESSOR-01",
        "fault_type": "valve_leak",
        "start_day": 12,
        "duration_hours": 24,
        "ramp": {
            "vibration_rms": 2.0,
            "temperature_c": 1.40,
            "pressure_bar":  0.75,  # pressure drops significantly
            "current_amp":   1.30,
        },
    },
    {
        "machine_id": "MOTOR-01",
        "fault_type": "overheating",
        "start_day": 20,
        "duration_hours": 36,
        "ramp": {
            "vibration_rms": 1.8,
            "temperature_c": 1.55,  # temperature spikes
            "pressure_bar":  1.0,
            "current_amp":   1.45,
        },
    },
]


# --- Generator -----------------------------------------------------------------

def _fault_multiplier(fault: dict, ts: datetime, base_dt: datetime) -> dict | None:
    """
    Returns per-sensor multipliers if `ts` falls inside a fault window,
    with a linear ramp-up. Returns None if outside window.
    """
    start = base_dt + timedelta(days=fault["start_day"])
    end   = start  + timedelta(hours=fault["duration_hours"])

    if not (start <= ts <= end):
        return None

    # Linear ramp: 0→1 over first 25% of window, then holds
    elapsed  = (ts - start).total_seconds()
    ramp_end = fault["duration_hours"] * 3600 * 0.25
    factor   = min(elapsed / ramp_end, 1.0)

    mults = {}
    for sensor, target in fault["ramp"].items():
        mults[sensor] = 1.0 + (target - 1.0) * factor
    return mults


def generate(days: int = 30, freq_minutes: int = 5) -> pd.DataFrame:
    """
    Generate synthetic sensor readings for all machines.

    Args:
        days: Number of days to simulate.
        freq_minutes: Sampling interval in minutes.

    Returns:
        DataFrame with columns:
            timestamp, machine_id, machine_type,
            vibration_rms, temperature_c, pressure_bar, current_amp,
            fault_type (None = normal)
    """
    rng    = np.random.default_rng(seed=42)
    base   = datetime(2024, 1, 1, 0, 0, 0)
    steps  = int(days * 24 * 60 / freq_minutes)
    rows   = []

    for machine_id, cfg in MACHINES.items():
        normal   = cfg["normal"]
        # Find faults belonging to this machine
        my_faults = [f for f in FAULT_SCENARIOS if f["machine_id"] == machine_id]

        for step in range(steps):
            ts = base + timedelta(minutes=step * freq_minutes)

            # Check if any fault is active
            active_fault = None
            mults        = None
            for fault in my_faults:
                mults = _fault_multiplier(fault, ts, base)
                if mults:
                    active_fault = fault["fault_type"]
                    break

            row = {"timestamp": ts, "machine_id": machine_id,
                   "machine_type": cfg["type"], "fault_type": active_fault}

            for sensor, params in normal.items():
                # Base reading with Gaussian noise
                val = rng.normal(params["mean"], params["std"])
                # Apply fault multiplier if active
                if mults and sensor in mults:
                    val *= mults[sensor]
                # Clip to physically plausible range (no negatives)
                val = max(val, 0.0)
                row[sensor] = round(float(val), 4)

            rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(["timestamp", "machine_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# --- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sensor data")
    parser.add_argument("--days",   type=int, default=30,              help="Days to simulate (default 30)")
    parser.add_argument("--freq",   type=int, default=5,               help="Sampling interval in minutes (default 5)")
    parser.add_argument("--output", type=str, default="data/sensors.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Generating {args.days} days of sensor data at {args.freq}-min intervals...")
    df = generate(days=args.days, freq_minutes=args.freq)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    total     = len(df)
    anomalies = df[df["fault_type"].notna()]
    print(f"  Total readings : {total:,}")
    print(f"  Anomaly rows   : {len(anomalies):,} ({len(anomalies)/total*100:.1f}%)")
    print(f"  Machines       : {df['machine_id'].unique().tolist()}")
    print(f"  Fault types    : {anomalies['fault_type'].unique().tolist()}")
    print(f"  Saved to       : {out}")


if __name__ == "__main__":
    main()
