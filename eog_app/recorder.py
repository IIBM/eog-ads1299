"""Simple ADS1299 configuration, live signal display, and recorder."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eog_app.device import SerialReader
from eog_app.signal_viewer import SignalPlot

SUPPORTED_SAMPLE_RATES = (250, 500, 1000)
SUPPORTED_GAINS = (1, 2, 4, 6, 8, 12, 24)
DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "EOG_data"


class RecorderWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EOG Signal Recorder")
        self.resize(950, 690)
        self.reader: SerialReader | None = None
        self.capture_file: TextIO | None = None
        self.capture_writer: csv.writer | None = None
        self.capture_path: Path | None = None
        self.sample_count = 0
        self.active_gain = 24

        root = QWidget(self)
        layout = QVBoxLayout(root)
        title = QLabel("EOG SIGNAL RECORDER")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        settings = QFormLayout()
        self.port_input = QLineEdit("/dev/ttyUSB0")
        settings.addRow("Serial port", self.port_input)

        self.sample_rate_input = QComboBox()
        self.sample_rate_input.addItems(
            [str(rate) for rate in SUPPORTED_SAMPLE_RATES]
        )
        self.sample_rate_input.setCurrentText("250")
        settings.addRow("ADS1299 sample rate", self.sample_rate_input)

        self.gain_input = QComboBox()
        self.gain_input.addItems([str(gain) for gain in SUPPORTED_GAINS])
        self.gain_input.setCurrentText("24")
        settings.addRow("ADS1299 channel gain", self.gain_input)

        storage_row = QWidget()
        storage_layout = QHBoxLayout(storage_row)
        storage_layout.setContentsMargins(0, 0, 0, 0)
        self.directory_input = QLineEdit(str(DEFAULT_DATA_DIRECTORY))
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._choose_directory)
        storage_layout.addWidget(self.directory_input, 1)
        storage_layout.addWidget(browse_button)
        settings.addRow("Save recordings to", storage_row)
        layout.addLayout(settings)

        self.status_label = QLabel(
            "Choose the acquisition settings and connect. Recording starts after "
            "the Arduino confirms its configuration."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.signal_plot = SignalPlot(
            sample_rate=250,
            channel_labels=("CH1 differential (µV)", "CH2 differential (µV)"),
        )
        layout.addWidget(self.signal_plot, 1)

        self.connect_button = QPushButton("Connect, configure & record")
        self.connect_button.clicked.connect(self._toggle_connection)
        layout.addWidget(self.connect_button)
        self.setCentralWidget(root)

    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose recording directory",
            self.directory_input.text(),
        )
        if directory:
            self.directory_input.setText(directory)

    def _toggle_connection(self) -> None:
        if self.reader is not None:
            self._disconnect()
            return

        port = self.port_input.text().strip()
        directory = Path(self.directory_input.text()).expanduser()
        if not port:
            QMessageBox.warning(self, "Missing serial port", "Enter a serial port path.")
            return
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.capture_path = directory / (
                datetime.now().astimezone().strftime("eog_%Y%m%d_%H%M%S_%f.csv")
            )
            self.capture_file = self.capture_path.open(
                "w", newline="", encoding="utf-8"
            )
            self.capture_writer = csv.writer(self.capture_file)
            self.capture_writer.writerow(
                (
                    "elapsed_seconds",
                    "channel_1_mV",
                    "channel_2_mV",
                    "channel_1_differential_uV",
                    "channel_2_differential_uV",
                )
            )
            self.capture_file.flush()
        except OSError as error:
            self._close_capture()
            QMessageBox.critical(
                self,
                "Cannot create recording",
                f"Could not open the selected save location:\n{error}",
            )
            return

        sample_rate = int(self.sample_rate_input.currentText())
        self.active_gain = int(self.gain_input.currentText())
        self.sample_count = 0
        self.signal_plot.samples.clear()
        self.signal_plot.set_sample_rate(sample_rate)
        self.reader = SerialReader(
            port,
            sample_rate=sample_rate,
            gain=self.active_gain,
        )
        self.reader.samples_received.connect(self._receive_samples)
        self.reader.status_changed.connect(self._reader_status)
        self.reader.error_occurred.connect(self._reader_error)
        self.reader.finished.connect(self._reader_finished)
        self.reader.start()
        self._set_connected_ui(True)
        self.status_label.setText(f"Waiting for Arduino on {port}…")

    def _reader_status(self, message: str) -> None:
        if self.reader is not None:
            self.status_label.setText(message)

    def _receive_samples(self, samples: object) -> None:
        values = np.asarray(samples, dtype=np.float64)
        differential_uv = values * 1000.0
        self.signal_plot.add_samples(differential_uv)

        if self.capture_writer is None or self.capture_file is None:
            return
        sample_rate = int(self.sample_rate_input.currentText())
        first_sample = self.sample_count
        rows = (
            (
                f"{(first_sample + index) / sample_rate:.6f}",
                f"{row[0]:.9g}",
                f"{row[1]:.9g}",
                f"{differential_uv[index, 0]:.9g}",
                f"{differential_uv[index, 1]:.9g}",
            )
            for index, row in enumerate(values)
        )
        try:
            self.capture_writer.writerows(rows)
            self.sample_count += len(values)
            self.capture_file.flush()
        except OSError as error:
            self.status_label.setText(f"Recording write failed: {error}")
            QMessageBox.critical(
                self,
                "Recording write failed",
                f"Could not save data to {self.capture_path}:\n{error}",
            )
            self._disconnect()
            return
        self.status_label.setText(
            f"Recording {sample_rate} SPS, gain {self.active_gain} · "
            f"{self.sample_count:,} samples · {self.capture_path}"
        )

    def _reader_error(self, message: str) -> None:
        self.status_label.setText(f"Serial/configuration error: {message}")
        QMessageBox.critical(self, "Serial/configuration error", message)
        self._close_capture()
        self._set_connected_ui(False)

    def _reader_finished(self) -> None:
        if self.reader is not None and not self.reader.isRunning():
            self.reader = None
            self._close_capture()
            self._set_connected_ui(False)

    def _disconnect(self) -> None:
        reader = self.reader
        self.reader = None
        if reader is not None:
            reader.stop()
        self._close_capture()
        self._set_connected_ui(False)
        if self.capture_path is not None and self.sample_count:
            self.status_label.setText(
                f"Recording saved: {self.capture_path} "
                f"({self.sample_count:,} samples)"
            )
        else:
            self.status_label.setText("Disconnected; no samples were recorded.")

    def _close_capture(self) -> None:
        if self.capture_file is not None:
            self.capture_file.close()
        self.capture_file = None
        self.capture_writer = None

    def _set_connected_ui(self, connected: bool) -> None:
        self.connect_button.setText(
            "Disconnect & finish recording"
            if connected
            else "Connect, configure & record"
        )
        for control in (
            self.port_input,
            self.sample_rate_input,
            self.gain_input,
            self.directory_input,
        ):
            control.setEnabled(not connected)

    def closeEvent(self, event: Any) -> None:
        if self.reader is not None:
            reader = self.reader
            self.reader = None
            reader.stop()
        self._close_capture()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = RecorderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
