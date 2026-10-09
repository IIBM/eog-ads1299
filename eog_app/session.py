"""Portable session-file management for calibration and live tests."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def create_session_dir(data_root: Path) -> Path:
    """Create a uniquely timestamped directory for one EOG run."""
    session_dir = data_root / datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H-%M-%S_%f"
    )
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def save_raw_samples(path: Path, samples: NDArray[np.generic]) -> None:
    values = np.asarray(samples, dtype=np.float64)
    np.savetxt(path, values, delimiter=",", header="channel_1,channel_2", comments="")


def save_calibration(
    session_dir: Path,
    features: NDArray[np.generic],
    positions: NDArray[np.generic],
    events: list[dict[str, Any]],
) -> None:
    np.savetxt(
        session_dir / "calibration_features.csv",
        np.asarray(features, dtype=np.float64),
        delimiter=",",
        header="channel_1,channel_2",
        comments="",
    )
    np.savetxt(
        session_dir / "calibration_positions.csv",
        np.asarray(positions, dtype=np.float64),
        delimiter=",",
        header="x,y",
        comments="",
    )
    with (session_dir / "calibration_events.json").open("w", encoding="utf-8") as file:
        json.dump(events, file, indent=2)


def save_predictions(
    path: Path,
    rows: list[tuple[float, float, float, float, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("elapsed_seconds", "x", "y", "double_blink", "target"))
        writer.writerows(rows)


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
