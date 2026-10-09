import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv
import io

import numpy as np
from PySide6.QtWidgets import QApplication

from eog_app.device import SerialReader
from eog_app.recorder import SUPPORTED_GAINS, SUPPORTED_SAMPLE_RATES, RecorderWindow


class ReadySerial:
    def __init__(self) -> None:
        self.commands: list[bytes] = []

    def write(self, command: bytes) -> None:
        self.commands.append(command)

    def readline(self) -> bytes:
        return b"READY,1000,12\n"


def test_configured_serial_reader_sends_settings_and_waits_for_ack() -> None:
    reader = SerialReader("/dev/ttyUSB0", sample_rate=1000, gain=12)
    connection = ReadySerial()

    reader._configure_device(connection)

    assert connection.commands == [b"CFG,1000,12\n"]


def test_recorder_saves_millivolts_and_microvolts() -> None:
    app = QApplication.instance() or QApplication([])
    window = RecorderWindow()
    capture = io.StringIO()
    window.capture_file = capture
    window.capture_writer = csv.writer(capture)
    window.capture_writer.writerow(
        (
            "elapsed_seconds",
            "channel_1_mV",
            "channel_2_mV",
            "channel_1_differential_uV",
            "channel_2_differential_uV",
        )
    )
    window.active_gain = 12
    window.sample_rate_input.setCurrentText("1000")

    window._receive_samples(np.array([[12.0, -24.0]]))

    rows = list(csv.reader(io.StringIO(capture.getvalue())))
    assert rows[1] == ["0.000000", "12", "-24", "12000", "-24000"]
    assert list(window.signal_plot.samples) == [(12000.0, -24000.0)]
    window.close()
    del app


def test_recorder_offers_supported_device_settings() -> None:
    assert SUPPORTED_SAMPLE_RATES == (250, 500, 1000)
    assert SUPPORTED_GAINS == (1, 2, 4, 6, 8, 12, 24)
