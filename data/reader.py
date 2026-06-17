"""
Data reader utility — thin wrapper around store.py.

Returns clean, typed data structures that the agent and model both consume.
Keeping this separate from store.py means the model/agent don't import
SQLite logic directly — easy to swap backends later.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from data.store import get_readings, get_latest_reading, get_all_machine_ids

SENSOR_COLS = ["vibration_rms", "temperature_c", "pressure_bar", "current_amp"]


@dataclass
class SensorWindow:
    """A time-series window of sensor readings for one machine."""
    machine_id:   str
    machine_type: str
    timestamps:   list[datetime]
    readings:     dict[str, list[float]]   # sensor_name → list of values
    fault_labels: list[str | None]         # ground-truth labels (None = normal)

    @property
    def n_samples(self) -> int:
        return len(self.timestamps)

    @property
    def as_array(self) -> np.ndarray:
        """Shape: (n_samples, n_sensors). Column order = SENSOR_COLS."""
        return np.array([self.readings[s] for s in SENSOR_COLS], dtype=np.float32).T

    def latest_values(self) -> dict[str, float]:
        """Most recent reading as a flat dict."""
        return {s: self.readings[s][-1] for s in SENSOR_COLS}

    def summary_stats(self) -> dict[str, dict[str, float]]:
        """Mean, std, min, max per sensor — useful for the agent's context."""
        stats = {}
        for sensor in SENSOR_COLS:
            arr = np.array(self.readings[sensor])
            stats[sensor] = {
                "mean": round(float(arr.mean()), 4),
                "std":  round(float(arr.std()),  4),
                "min":  round(float(arr.min()),  4),
                "max":  round(float(arr.max()),  4),
            }
        return stats


def load_window(machine_id: str, last_n: int = 50) -> SensorWindow:
    """
    Load the most recent `last_n` readings for a machine.

    Args:
        machine_id: e.g. "PUMP-01"
        last_n: window size (50 readings × 5-min intervals = ~4 hours)

    Returns:
        SensorWindow ready for model inference or agent context.
    """
    df = get_readings(machine_id, last_n=last_n)

    if df.empty:
        raise ValueError(f"No data found for machine '{machine_id}'.")

    return SensorWindow(
        machine_id=machine_id,
        machine_type=str(df["machine_type"].iloc[0]),
        timestamps=[t.to_pydatetime() for t in df["timestamp"]],
        readings={s: df[s].tolist() for s in SENSOR_COLS},
        fault_labels=df["fault_type"].tolist(),
    )


def load_all_latest() -> dict[str, dict]:
    """
    Return the single most recent reading for every machine.
    Useful for the dashboard overview page.

    Returns:
        { machine_id: {sensor: value, ...} }
    """
    result = {}
    for mid in get_all_machine_ids():
        row = get_latest_reading(mid)
        if row:
            result[mid] = row
    return result


def sensor_context_string(window: SensorWindow) -> str:
    """
    Format a SensorWindow as a human-readable string for the agent's prompt.

    Example output:
        Machine: PUMP-01 (centrifugal_pump) | Last 50 readings (~4 hours)
        vibration_rms : mean=2.31  std=0.82  min=2.10  max=6.45  latest=6.45
        temperature_c : mean=66.2  std=2.1   min=64.5  max=79.8  latest=79.8
        ...
    """
    lines = [
        f"Machine: {window.machine_id} ({window.machine_type})"
        f" | Last {window.n_samples} readings",
        f"Period : {window.timestamps[0].strftime('%Y-%m-%d %H:%M')}"
        f" → {window.timestamps[-1].strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    stats   = window.summary_stats()
    latest  = window.latest_values()
    for sensor in SENSOR_COLS:
        s = stats[sensor]
        lines.append(
            f"  {sensor:<18} mean={s['mean']:<7} std={s['std']:<6}"
            f" min={s['min']:<7} max={s['max']:<7} latest={latest[sensor]}"
        )
    return "\n".join(lines)


# --- Quick smoke-test ----------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    for mid in get_all_machine_ids():
        print(f"\n{'='*60}")
        window = load_window(mid, last_n=50)
        print(sensor_context_string(window))
        print(f"  Array shape : {window.as_array.shape}")
