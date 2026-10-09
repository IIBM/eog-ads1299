import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from eog_app.app import EOGWindow
from eog_app.signal_viewer import SignalPlot


def test_signal_plot_keeps_one_second_of_both_raw_channels() -> None:
    app = QApplication.instance() or QApplication([])
    plot = SignalPlot(sample_rate=10)

    plot.add_samples(np.arange(120, dtype=np.float64).reshape(60, 2))

    assert len(plot.samples) == 10
    assert plot.samples[0] == (100.0, 101.0)
    assert plot.samples[-1] == (118.0, 119.0)
    image = QImage(plot.size(), QImage.Format.Format_ARGB32)
    plot.render(image)
    del app


def test_main_window_opens_viewer_and_forwards_new_samples() -> None:
    app = QApplication.instance() or QApplication([])
    window = EOGWindow()
    values = np.array([[1.25, -2.5], [3.0, 4.0]])

    window._open_signal_viewer()
    window._receive_samples(values)

    assert window.signal_viewer is not None
    assert list(window.signal_viewer.plot.samples) == [
        (1.25, -2.5),
        (3.0, 4.0),
    ]
    window.close()
    del app


def test_signal_plot_rejects_wrong_channel_shape() -> None:
    app = QApplication.instance() or QApplication([])
    plot = SignalPlot()

    with pytest.raises(ValueError, match="shape"):
        plot.add_samples(np.array([1.0, 2.0]))
    del app
