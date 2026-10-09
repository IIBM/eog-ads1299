"""Signal-processing routines for the ADS1299 EOG stream."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def split_channels(
    samples: NDArray[np.generic],
    active_channels: int = 2,
) -> FloatArray:
    """Convert interleaved millivolt samples into microvolts by channel."""
    values = np.asarray(samples, dtype=np.float64)
    if active_channels < 1:
        raise ValueError("active_channels must be positive")

    if values.ndim == 2:
        if values.shape[1] != active_channels:
            raise ValueError(
                f"expected {active_channels} channels, got {values.shape[1]}"
            )
        return values * 1000.0

    values = values.reshape(-1)
    if values.size % active_channels:
        raise ValueError("sample count is not divisible by active_channels")
    return values.reshape(-1, active_channels) * 1000.0


def extract_features(
    samples: NDArray[np.generic],
    *,
    sample_rate: int = 250,
    derivative_samples: int = 50,
    noise_floor: tuple[float, float] = (100.0, 80.0),
    blink_threshold: float | None = None,
) -> tuple[FloatArray, bool]:
    """Extract channel peak changes and detect a rapid double blink."""
    if sample_rate <= 0 or derivative_samples <= 0:
        raise ValueError("sample_rate and derivative_samples must be positive")

    channels = split_channels(samples, 2)
    if channels.shape[0] <= derivative_samples:
        return np.zeros(2, dtype=np.float64), False

    derivative = channels[derivative_samples:] - channels[:-derivative_samples]
    features = np.zeros(2, dtype=np.float64)
    double_blink = False

    for channel_index, floor in enumerate(noise_floor):
        signal = derivative[:, channel_index]
        filtered = np.where(np.abs(signal) > floor, signal, 0.0)
        positive_peak = float(np.max(filtered))
        negative_peak = float(np.min(filtered))

        if blink_threshold is not None and (
            positive_peak > blink_threshold
            or abs(negative_peak) > blink_threshold
        ):
            high_peaks = np.flatnonzero(
                (signal[1:-1] > signal[:-2])
                & (signal[1:-1] >= signal[2:])
                & (signal[1:-1] > blink_threshold)
            ) + 1
            if high_peaks.size >= 2 and np.min(np.diff(high_peaks)) < sample_rate / 2:
                double_blink = True
                positive_peak = 0.0
                negative_peak = 0.0

        feature = positive_peak + negative_peak
        features[channel_index] = 0.0 if abs(feature) < floor else feature

    return features, double_blink


def augment_calibration_data(
    features: NDArray[np.generic],
    positions: NDArray[np.generic],
    *,
    points_per_sample: int = 39,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Expand calibration examples with bounded noise and shuffle them."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(positions, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 2 or y.shape[1] != 2:
        raise ValueError("features and positions must both have shape (N, 2)")
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("at least two paired calibration samples are required")
    if points_per_sample < 0:
        raise ValueError("points_per_sample cannot be negative")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("calibration data must contain only finite values")

    rng = np.random.default_rng(seed)
    limits = np.maximum(np.max(np.abs(x), axis=0), 1.0) * np.array([0.9 / 4, 0.9 / 2])
    noise = rng.uniform(
        -limits,
        limits,
        size=(x.shape[0], points_per_sample, 2),
    )
    augmented_x = np.concatenate((x[:, None, :], x[:, None, :] + noise), axis=1)
    augmented_y = np.repeat(y[:, None, :], points_per_sample + 1, axis=1)
    augmented_x = augmented_x.reshape(-1, 2)
    augmented_y = augmented_y.reshape(-1, 2)
    order = rng.permutation(augmented_x.shape[0])
    return augmented_x[order], augmented_y[order]
