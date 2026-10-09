"""Serial reader for the Arduino's continuous two-channel float stream."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal


class SerialReader(QThread):
    """Read and decode pairs of little-endian float32 samples."""

    samples_received = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        port: str,
        *,
        baud_rate: int = 500000,
        channels: int = 2,
        sample_rate: int | None = None,
        gain: int | None = None,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.port = port
        self.baud_rate = baud_rate
        self.channels = channels
        self.sample_rate = sample_rate
        self.gain = gain
        self._running = True

        if (sample_rate is None) != (gain is None):
            raise ValueError("sample_rate and gain must be configured together")

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    @staticmethod
    def _alignment_offset(data: bytes, channels: int) -> int | None:
        frame_size = channels * 4
        best_offset: int | None = None
        best_score = 0.0
        for offset in range(frame_size):
            usable = (len(data) - offset) // 4
            if usable < 64:
                continue
            values = np.frombuffer(data, dtype="<f4", count=usable, offset=offset)
            valid = np.isfinite(values) & (np.abs(values) <= 4500.0)
            score = float(np.mean(valid))
            nonzero_score = float(np.mean(np.abs(values[valid]) > 1e-12)) if valid.any() else 0.0
            score = score * nonzero_score
            if score > best_score:
                best_score = score
                best_offset = offset
        return best_offset if best_score >= 0.98 else None

    def run(self) -> None:
        try:
            import serial

            with serial.Serial(
                self.port,
                self.baud_rate,
                timeout=0.5,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            ) as connection:
                if self.sample_rate is not None and self.gain is not None:
                    self._configure_device(connection)
                else:
                    self.status_changed.emit(
                        "Connected; synchronizing to the Arduino float stream…"
                    )
                pending = bytearray()
                alignment: int | None = None
                sync_deadline = time.monotonic() + 6.0

                while self._running:
                    chunk = connection.read(400)
                    if not chunk:
                        if alignment is None and time.monotonic() > sync_deadline:
                            raise TimeoutError(
                                "No valid ADS1299 float stream received within 6 seconds. "
                                "Check the Arduino firmware and serial port."
                            )
                        continue

                    pending.extend(chunk)
                    if alignment is None:
                        alignment = self._alignment_offset(bytes(pending), self.channels)
                        if alignment is None:
                            if len(pending) > 8192:
                                del pending[:-8192]
                            continue
                        del pending[:alignment]
                        self.status_changed.emit(
                            "Connected; receiving two-channel EOG samples."
                        )

                    frame_size = self.channels * 4
                    byte_count = len(pending) - (len(pending) % frame_size)
                    if byte_count:
                        values = np.frombuffer(
                            bytes(pending[:byte_count]), dtype="<f4"
                        ).reshape(-1, self.channels)
                        del pending[:byte_count]
                        if values.size and (
                            not np.isfinite(values).all()
                            or np.any(np.abs(values) > 4500.0)
                        ):
                            alignment = None
                            pending = bytearray(pending)
                            continue
                        self.samples_received.emit(values.astype(np.float64))
        except Exception as error:
            if self._running:
                self.error_occurred.emit(str(error))
        finally:
            self.status_changed.emit("Serial reader stopped.")

    def _configure_device(self, connection: Any) -> None:
        self.status_changed.emit(
            f"Connected; configuring ADS1299 for {self.sample_rate} SPS, "
            f"gain {self.gain}…"
        )
        command = f"CFG,{self.sample_rate},{self.gain}\n".encode("ascii")
        deadline = time.monotonic() + 12.0
        while self._running and time.monotonic() < deadline:
            connection.write(command)
            response = connection.readline().decode("ascii", errors="replace").strip()
            if response == f"READY,{self.sample_rate},{self.gain}":
                self.status_changed.emit(
                    f"Streaming at {self.sample_rate} SPS, ADS1299 gain {self.gain}."
                )
                return
            if response.startswith("ERROR,"):
                raise RuntimeError(f"Arduino rejected the configuration: {response[6:]}")
        if self._running:
            raise TimeoutError(
                "Arduino did not acknowledge the configuration within 12 seconds. "
                "Flash the updated Arduino sketch, then reconnect."
            )
