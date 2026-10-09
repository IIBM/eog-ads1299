import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from eog_app.app import EOGWindow
from eog_app.model import train_model


def test_calibration_prompt_highlights_requested_control() -> None:
    app = QApplication.instance() or QApplication([])
    window = EOGWindow()
    window.calibration_sequence = [(0.5, 0.825, "Look up", 1)]
    window.calibration_index = 0

    window._show_calibration_prompt()

    assert window.prompt_label.text() == "Look up"
    assert "#8fd3a8" in window.control_tiles["forward"].styleSheet()
    window.close()
    del app


def test_live_predictions_update_from_buffered_sample_windows(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    window = EOGWindow()
    window.model_path = tmp_path / "model.npz"
    train_model(
        np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]]),
        np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]]),
        window.model_path,
        epochs=5,
    )
    window.active_stage = "test"
    window.test_targets = [(0.5, 0.825, "Forward", "right.mp3")]
    highlighted: list[tuple[float, float, bool]] = []
    window._highlight_control = lambda x, y, selected: highlighted.append(
        (x, y, selected)
    )
    first = np.zeros((250, 2), dtype=np.float64)
    first[100:, 0] = 10.0

    window._receive_samples(first[:50])
    window._receive_samples(first[50:])
    window._receive_samples(np.full((250, 2), 10.0))

    assert len(highlighted) == 2
    assert "x=" in window.prediction_label.text()
    assert len(window.prediction_rows) == 2
    window.close()
    del app
