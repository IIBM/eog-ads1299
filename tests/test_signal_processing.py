import numpy as np
import pytest

from eog_app.signal_processing import (
    augment_calibration_data,
    extract_features,
    split_channels,
)


def test_split_channels_deinterleaves_millivolts_to_microvolts() -> None:
    samples = np.array([1.0, 2.0, 3.0, 4.0])

    result = split_channels(samples)

    np.testing.assert_array_equal(result, [[1000.0, 2000.0], [3000.0, 4000.0]])


def test_split_channels_rejects_partial_frame() -> None:
    with pytest.raises(ValueError, match="divisible"):
        split_channels(np.array([1.0, 2.0, 3.0]))


def test_extract_features_detects_channel_change() -> None:
    samples = np.zeros((20, 2), dtype=np.float64)
    samples[8:, 0] = 0.125

    features, double_blink = extract_features(
        samples,
        sample_rate=10,
        derivative_samples=2,
        noise_floor=(100.0, 80.0),
    )

    assert features[0] == pytest.approx(125.0)
    assert features[1] == 0.0
    assert not double_blink


def test_augment_calibration_data_preserves_examples_and_targets() -> None:
    features = np.array([[10.0, -20.0], [30.0, 40.0]])
    positions = np.array([[0.1, 0.5], [0.9, 0.5]])

    expanded_x, expanded_y = augment_calibration_data(
        features, positions, points_per_sample=3, seed=123
    )

    assert expanded_x.shape == (8, 2)
    assert expanded_y.shape == (8, 2)
    for position in positions:
        assert np.any(np.all(expanded_y == position, axis=1))
