"""Desktop calibration and live-test application for the EOG BCI."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eog_app.device import SerialReader
from eog_app.model import predict, train_model
from eog_app.session import (
    create_session_dir,
    save_calibration,
    save_metadata,
    save_predictions,
    save_raw_samples,
)
from eog_app.signal_viewer import SignalViewerWindow
from eog_app.signal_processing import augment_calibration_data, extract_features

DATA_ROOT = Path(__file__).resolve().parents[1] / "EOG_data"
ASSET_ROOT = Path(__file__).resolve().parents[1] / "Matlab" / "UI controls - NN"
CALIBRATION_POINTS = (
    (0.50, 0.825, "Look up"),
    (0.90, 0.495, "Look right"),
    (0.50, 0.165, "Look down"),
    (0.10, 0.495, "Look left"),
    (0.50, 0.495, "Look center"),
)
TEST_TARGETS = (
    (0.50, 0.825, "Forward", "right.mp3"),
    (0.90, 0.495, "Turn right", "backward.mp3"),
    (0.50, 0.165, "Backward", "left.mp3"),
    (0.10, 0.495, "Turn left", "forward.mp3"),
)


class EOGWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EOG BCI — Calibration and Live Test")
        self.resize(1050, 700)
        self.reader: SerialReader | None = None
        self.session_dir: Path | None = None
        self.model_path: Path | None = None
        self.active_stage: str | None = None
        self.stream_samples: list[np.ndarray] = []
        self.stage_samples: list[np.ndarray] = []
        self.interval_samples: list[np.ndarray] = []
        self.live_pending_samples = np.empty((0, 2), dtype=np.float64)
        self.live_context = np.empty((0, 2), dtype=np.float64)
        self.calibration_features: list[np.ndarray] = []
        self.calibration_positions: list[tuple[float, float]] = []
        self.calibration_events: list[dict[str, Any]] = []
        self.prediction_rows: list[tuple[float, float, float, float, str]] = []
        self.zero_drift = np.zeros(2, dtype=np.float64)
        self.accumulated_features = np.zeros(2, dtype=np.float64)
        self.calibration_sequence: list[tuple[float, float, str, int]] = []
        self.calibration_index = 0
        self.elapsed_stage = 0.0
        self.test_index = 0
        self.test_targets: list[tuple[float, float, str, str]] = []
        self.last_prediction = np.array([0.5, 0.5], dtype=np.float64)
        self.control_tiles: dict[str, QLabel] = {}
        self.signal_viewer: SignalViewerWindow | None = None
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.audio_player = QMediaPlayer(self)
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_player.errorOccurred.connect(self._audio_error)
        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self._stage_tick)
        self.test_clock = QTimer(self)
        self.test_clock.setInterval(250)
        self.test_clock.timeout.connect(self._update_elapsed)
        self.stage_started = 0.0

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)

        title = QLabel("EOG CONTROL")
        title.setStyleSheet("font-size: 26px; font-weight: bold;")
        layout.addWidget(title)

        connection = QGroupBox("Device connection")
        connection_layout = QHBoxLayout(connection)
        self.port_input = QLineEdit("/dev/ttyUSB0")
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._toggle_connection)
        connection_layout.addWidget(QLabel("Serial port"))
        connection_layout.addWidget(self.port_input, 1)
        connection_layout.addWidget(self.connect_button)
        layout.addWidget(connection)

        config = QGroupBox("Acquisition settings")
        config_layout = QFormLayout(config)
        self.sample_rate_input = QSpinBox()
        self.sample_rate_input.setRange(50, 2000)
        self.sample_rate_input.setValue(250)
        self.gain_input = QSpinBox()
        self.gain_input.setRange(1, 1000)
        self.gain_input.setValue(24)
        self.interval_input = QSpinBox()
        self.interval_input.setRange(2, 30)
        self.interval_input.setValue(4)
        self.test_interval_input = QSpinBox()
        self.test_interval_input.setRange(2, 30)
        self.test_interval_input.setValue(8)
        self.stimuli_input = QSpinBox()
        self.stimuli_input.setRange(4, 1000)
        self.stimuli_input.setValue(30)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(100, 100000)
        self.epochs_input.setSingleStep(100)
        self.epochs_input.setValue(3000)
        config_layout.addRow("Sample rate (Hz)", self.sample_rate_input)
        config_layout.addRow("Arduino gain", self.gain_input)
        config_layout.addRow("Calibration prompt interval (s)", self.interval_input)
        config_layout.addRow("Test prompt interval (s)", self.test_interval_input)
        config_layout.addRow("Test stimuli", self.stimuli_input)
        config_layout.addRow("Training epochs", self.epochs_input)
        layout.addWidget(config)

        self.status_label = QLabel("Connect to the Arduino to begin.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.signal_viewer_button = QPushButton("Open live signal viewer")
        self.signal_viewer_button.clicked.connect(self._open_signal_viewer)
        layout.addWidget(self.signal_viewer_button)

        self.prompt_label = QLabel("Ready")
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setMinimumHeight(130)
        self.prompt_label.setStyleSheet(
            "font-size: 34px; font-weight: bold; background: #202d3b; "
            "color: white; border-radius: 10px;"
        )
        layout.addWidget(self.prompt_label)

        self.prediction_label = QLabel("Predicted position: center")
        self.prediction_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.prediction_label)
        self.raw_label = QLabel("Waiting for samples")
        layout.addWidget(self.raw_label)

        actions = QHBoxLayout()
        self.calibrate_button = QPushButton("Start calibration")
        self.calibrate_button.clicked.connect(self._start_calibration)
        self.test_button = QPushButton("Start live test")
        self.test_button.clicked.connect(self._start_test)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_stage)
        self.test_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        actions.addWidget(self.calibrate_button)
        actions.addWidget(self.test_button)
        actions.addWidget(self.stop_button)
        layout.addLayout(actions)

        controls = QGridLayout()
        for key, text, row, column in (
            ("forward", "Forward", 0, 1),
            ("left", "Turn left", 1, 0),
            ("center", "Recalibrate", 1, 1),
            ("right", "Turn right", 1, 2),
            ("backward", "Backward", 2, 1),
        ):
            tile = QLabel(text)
            tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tile.setMinimumHeight(56)
            self.control_tiles[key] = tile
            tile.setStyleSheet(
                "background: #d9e2ec; border: 2px solid #829ab1; "
                "border-radius: 6px; font-size: 16px;"
            )
            controls.addWidget(tile, row, column)
        self._highlight_control(0.5, 0.495, False)
        layout.addLayout(controls)

        self.setCentralWidget(root)

    def _toggle_connection(self) -> None:
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
            self.connect_button.setText("Connect")
            self.calibrate_button.setEnabled(False)
            self.test_button.setEnabled(False)
            self.status_label.setText("Disconnected.")
            return

        port = self.port_input.text().strip()
        if not port:
            QMessageBox.warning(self, "Missing serial port", "Enter a serial port path.")
            return
        self.reader = SerialReader(
            port,
            sample_rate=self.sample_rate_input.value(),
            gain=self.gain_input.value(),
        )
        self.reader.samples_received.connect(self._receive_samples)
        self.reader.status_changed.connect(self.status_label.setText)
        self.reader.error_occurred.connect(self._reader_error)
        self.reader.finished.connect(self._reader_finished)
        self.reader.start()
        self.connect_button.setText("Disconnect")
        self.calibrate_button.setEnabled(False)
        self.test_button.setEnabled(False)

    def _reader_finished(self) -> None:
        if self.reader is not None and not self.reader.isRunning():
            self.reader = None
            self.connect_button.setText("Connect")
            self.calibrate_button.setEnabled(False)
            self.test_button.setEnabled(False)

    def _reader_error(self, message: str) -> None:
        self.status_label.setText(f"Serial error: {message}")
        QMessageBox.critical(self, "Serial communication error", message)

    def _receive_samples(self, samples: object) -> None:
        values = np.asarray(samples, dtype=np.float64)
        if self.signal_viewer is not None:
            self.signal_viewer.add_samples(values)
        self.stream_samples.append(values)
        if self.active_stage is not None:
            self.interval_samples.append(values)
            self.stage_samples.append(values)
        if self.active_stage == "test":
            self._process_live_samples(values)
        if len(self.stream_samples) > 1000:
            del self.stream_samples[:500]
        if self.reader is not None and self.reader.isRunning():
            self.calibrate_button.setEnabled(self.active_stage is None)
            self.test_button.setEnabled(
                self.active_stage is None and self.model_path is not None
            )
        flattened = np.concatenate(self.stream_samples[-20:], axis=0)
        self.raw_label.setText(
            f"Recent raw channel values: "
            f"{np.mean(flattened[:, 0]):.3f}, {np.mean(flattened[:, 1]):.3f} "
            f"— SD {np.std(flattened[:, 0]):.3f}, {np.std(flattened[:, 1]):.3f}"
        )

    def _open_signal_viewer(self) -> None:
        if self.signal_viewer is None:
            self.signal_viewer = SignalViewerWindow(
                sample_rate=self.sample_rate_input.value()
            )
            self.signal_viewer.destroyed.connect(self._signal_viewer_closed)
            if self.stream_samples:
                self.signal_viewer.add_samples(
                    np.concatenate(self.stream_samples, axis=0)
                )
        else:
            self.signal_viewer.plot.set_sample_rate(self.sample_rate_input.value())
        self.signal_viewer.show()
        self.signal_viewer.raise_()
        self.signal_viewer.activateWindow()

    def _signal_viewer_closed(self) -> None:
        self.signal_viewer = None

    def _start_calibration(self) -> None:
        if self.reader is None or not self.reader.isRunning():
            QMessageBox.warning(self, "Device not connected", "Connect to the Arduino first.")
            return
        self.session_dir = create_session_dir(DATA_ROOT)
        self.model_path = None
        self.calibration_features.clear()
        self.calibration_positions.clear()
        self.calibration_events.clear()
        center = (0.5, 0.495, "Look center")
        targets = list(CALIBRATION_POINTS)
        np.random.default_rng().shuffle(targets)
        self.calibration_sequence = []
        for index, point in enumerate(targets, start=1):
            self.calibration_sequence.extend(
                (
                    (*center, 0),
                    (*point, index),
                )
            )
        self.calibration_sequence.append((*center, 0))
        self.zero_drift = np.zeros(2, dtype=np.float64)
        self.calibration_index = 0
        self.stream_samples.clear()
        self._begin_stage("calibration")
        self._show_calibration_prompt()

    def _show_calibration_prompt(self) -> None:
        x, y, message, _ = self.calibration_sequence[self.calibration_index]
        self.prompt_label.setText(message)
        self._highlight_control(x, y, False)
        self.interval_samples.clear()
        self.status_label.setText(
            f"Calibration: prompt {self.calibration_index + 1} of "
            f"{len(self.calibration_sequence)}. Follow the displayed direction."
        )

    def _start_test(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            QMessageBox.warning(
                self,
                "No trained model",
                "Complete calibration and model training before the live test.",
            )
            return
        target_indexes = np.tile(
            np.arange(len(TEST_TARGETS)),
            int(np.ceil(self.stimuli_input.value() / len(TEST_TARGETS))),
        )[: self.stimuli_input.value()]
        np.random.default_rng().shuffle(target_indexes)
        self.test_targets = [TEST_TARGETS[index] for index in target_indexes]
        self.test_index = 0
        self.elapsed_stage = 0.0
        self.accumulated_features = np.zeros(2, dtype=np.float64)
        self.prediction_rows.clear()
        self.stream_samples.clear()
        self.live_pending_samples = np.empty((0, 2), dtype=np.float64)
        self.live_context = np.empty((0, 2), dtype=np.float64)
        self._begin_stage("test")
        self._show_test_prompt()

    def _show_test_prompt(self) -> None:
        if self.test_index >= len(self.test_targets):
            self._finish_test()
            return
        _, _, instruction, audio_file = self.test_targets[self.test_index]
        self.prompt_label.setText(instruction)
        self._play_audio_cue(audio_file)
        self.interval_samples.clear()
        self.status_label.setText(
            f"Live test: stimulus {self.test_index + 1} of {len(self.test_targets)}."
        )

    def _play_audio_cue(self, filename: str) -> None:
        audio_path = ASSET_ROOT / filename
        if not audio_path.is_file():
            self.status_label.setText(f"Audio cue not found: {audio_path}")
            return
        self.audio_player.stop()
        self.audio_player.setSource(QUrl.fromLocalFile(str(audio_path)))
        self.audio_player.play()

    def _audio_error(self, error: QMediaPlayer.Error, message: str) -> None:
        details = message or error.name
        self.status_label.setText(f"Audio playback error: {details}")

    def _begin_stage(self, stage: str) -> None:
        self.active_stage = stage
        self.stage_started = time.monotonic()
        self.interval_samples.clear()
        self.stage_samples.clear()
        self.tick_timer.start(
            (self.interval_input.value() if stage == "calibration" else self.test_interval_input.value())
            * 1000
        )
        self.test_clock.start()
        self.calibrate_button.setEnabled(False)
        self.test_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _update_elapsed(self) -> None:
        self.elapsed_stage = time.monotonic() - self.stage_started

    def _stage_tick(self) -> None:
        if self.active_stage == "test":
            self._record_test_interval()
            return
        if not self.interval_samples:
            self.status_label.setText(
                "No samples received during the prompt interval; check the device."
            )
            return
        samples = np.concatenate(self.interval_samples, axis=0)
        features, _ = extract_features(
            samples,
            sample_rate=self.sample_rate_input.value(),
        )
        if self.active_stage == "calibration":
            self._record_calibration_interval(features, samples)

    def _process_live_samples(self, samples: np.ndarray) -> None:
        self.live_pending_samples = np.concatenate(
            (self.live_pending_samples, samples), axis=0
        )
        block_size = self.sample_rate_input.value()
        derivative_samples = 50
        while self.live_pending_samples.shape[0] >= block_size:
            new_samples = self.live_pending_samples[:block_size]
            self.live_pending_samples = self.live_pending_samples[block_size:]
            window = np.concatenate((self.live_context, new_samples), axis=0)
            features, double_blink = extract_features(
                window,
                sample_rate=block_size,
                derivative_samples=derivative_samples,
                blink_threshold=500.0,
            )
            self.live_context = window[-derivative_samples:]
            self._update_live_prediction(features, double_blink)

    def _record_calibration_interval(
        self, features: np.ndarray, samples: np.ndarray
    ) -> None:
        x, y, message, control_index = self.calibration_sequence[self.calibration_index]
        is_center = control_index == 0
        if is_center:
            self.zero_drift = features
        else:
            self.calibration_features.append(features - self.zero_drift)
            self.calibration_positions.append((x, y))
        self.calibration_events.append(
            {
                "index": self.calibration_index,
                "x": x,
                "y": y,
                "prompt": message,
                "control_index": control_index,
                "feature_ch1": float(features[0]),
                "feature_ch2": float(features[1]),
                "sample_count": int(samples.shape[0]),
                "elapsed_seconds": round(time.monotonic() - self.stage_started, 3),
            }
        )
        self.calibration_index += 1
        if self.calibration_index >= len(self.calibration_sequence):
            self._finish_calibration()
        else:
            self._show_calibration_prompt()

    def _finish_calibration(self) -> None:
        self.tick_timer.stop()
        self.test_clock.stop()
        try:
            features, positions = augment_calibration_data(
                np.asarray(self.calibration_features),
                np.asarray(self.calibration_positions),
                seed=7,
            )
            assert self.session_dir is not None
            self.model_path = self.session_dir / "model.npz"
            save_calibration(
                self.session_dir,
                np.asarray(self.calibration_features),
                np.asarray(self.calibration_positions),
                self.calibration_events,
            )
            loss = train_model(
                features,
                positions,
                self.model_path,
                epochs=self.epochs_input.value(),
            )
            self._save_raw_capture("calibration_raw.csv")
            save_metadata(
                self.session_dir / "session.json",
                {
                    "serial_port": self.port_input.text().strip(),
                    "sample_rate_hz": self.sample_rate_input.value(),
                    "arduino_gain": self.gain_input.value(),
                    "training_epochs": self.epochs_input.value(),
                    "training_mse": loss,
                },
            )
            self.prompt_label.setText("Calibration complete")
            self.status_label.setText(
                f"Model trained (MSE {loss:.5f}). Session: {self.session_dir}"
            )
        except Exception as error:
            self.model_path = None
            self.prompt_label.setText("Calibration failed")
            self.status_label.setText(f"Calibration/training error: {error}")
            QMessageBox.critical(self, "Calibration failed", str(error))
        finally:
            self._finish_stage()

    def _update_live_prediction(
        self, features: np.ndarray, double_blink: bool
    ) -> None:
        self.accumulated_features += features - self.zero_drift
        assert self.model_path is not None
        self.last_prediction = predict(self.accumulated_features, self.model_path)[0]
        x, y = np.clip(self.last_prediction, 0.0, 1.0)
        self._highlight_control(float(x), float(y), double_blink)
        self.prediction_label.setText(
            f"Predicted position: x={x:.3f}, y={y:.3f}"
            + (" — double blink detected" if double_blink else "")
        )
        target = self.test_targets[self.test_index][2]
        self.prediction_rows.append(
            (
                time.monotonic() - self.stage_started,
                float(x),
                float(y),
                float(double_blink),
                target,
            )
        )
    def _record_test_interval(self) -> None:
        if not self.interval_samples:
            self.status_label.setText(
                "No samples received during the prompt interval; check the device."
            )
            return
        self.test_index += 1
        self._show_test_prompt()

    def _highlight_control(self, x: float, y: float, selected: bool) -> None:
        if y > 0.66 and 0.3 <= x <= 0.7:
            active = "forward"
        elif y < 0.33 and 0.3 <= x <= 0.7:
            active = "backward"
        elif x < 0.3 and 0.33 <= y <= 0.66:
            active = "left"
        elif x > 0.7 and 0.33 <= y <= 0.66:
            active = "right"
        elif 0.3 <= x <= 0.7 and 0.33 <= y <= 0.66:
            active = "center"
        else:
            active = ""
        for key, tile in self.control_tiles.items():
            if key == active:
                background = "#f6c453" if selected else "#8fd3a8"
                border = "#b42318" if selected else "#16803c"
                weight = "bold" if selected else "normal"
            else:
                background, border, weight = "#d9e2ec", "#829ab1", "normal"
            tile.setStyleSheet(
                f"background: {background}; border: 2px solid {border}; "
                f"border-radius: 6px; font-size: 16px; font-weight: {weight};"
            )

    def _finish_test(self) -> None:
        self.tick_timer.stop()
        self.test_clock.stop()
        if self.session_dir is not None:
            save_predictions(self.session_dir / "predictions.csv", self.prediction_rows)
            self._save_raw_capture("test_raw.csv")
        self.prompt_label.setText("Live test complete")
        self.status_label.setText(f"Results saved in {self.session_dir}")
        self._finish_stage()

    def _save_raw_capture(self, filename: str) -> None:
        if self.session_dir is not None and self.stage_samples:
            save_raw_samples(
                self.session_dir / filename,
                np.concatenate(self.stage_samples, axis=0),
            )

    def _stop_stage(self) -> None:
        self.tick_timer.stop()
        self.test_clock.stop()
        self.prompt_label.setText("Stopped")
        self.status_label.setText("The current stage was stopped.")
        self._finish_stage()

    def _finish_stage(self) -> None:
        self.active_stage = None
        self.stop_button.setEnabled(False)
        connected = self.reader is not None and self.reader.isRunning()
        self.calibrate_button.setEnabled(connected)
        self.test_button.setEnabled(connected and self.model_path is not None)

    def closeEvent(self, event: Any) -> None:
        self.tick_timer.stop()
        self.test_clock.stop()
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if self.signal_viewer is not None:
            self.signal_viewer.close()
            self.signal_viewer = None
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = EOGWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
