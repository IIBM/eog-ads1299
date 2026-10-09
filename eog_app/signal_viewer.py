"""Small scrolling viewer for the raw two-channel Arduino signal."""

from __future__ import annotations

from collections import deque

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class SignalPlot(QWidget):
    """Paint two rolling traces using the values received from the Arduino."""

    COLORS = (QColor("#45b7e8"), QColor("#f2a541"))
    WINDOW_SECONDS = 1

    def __init__(
        self,
        sample_rate: int = 250,
        parent: QWidget | None = None,
        channel_labels: tuple[str, str] = ("Channel 1", "Channel 2"),
    ) -> None:
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.channel_labels = channel_labels
        self.samples: deque[tuple[float, float]] = deque(
            maxlen=sample_rate * self.WINDOW_SECONDS
        )
        self.setMinimumSize(500, 360)
        self.setAutoFillBackground(False)

    def set_sample_rate(self, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self.samples = deque(
            self.samples,
            maxlen=sample_rate * self.WINDOW_SECONDS,
        )
        self.update()

    def add_samples(self, samples: object) -> None:
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("signal viewer expects samples with shape (N, 2)")
        if not np.isfinite(values).all():
            raise ValueError("signal viewer received non-finite samples")
        self.samples.extend((float(row[0]), float(row[1])) for row in values)
        self.update()

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111923"))

        margin_left = 74
        margin_right = 24
        margin_top = 18
        margin_bottom = 34
        gap = 30
        chart_height = max(
            60,
            (self.height() - margin_top - margin_bottom - gap) / 2,
        )
        chart_width = max(1, self.width() - margin_left - margin_right)
        chart_rects = (
            (margin_left, margin_top, chart_width, chart_height),
            (margin_left, margin_top + chart_height + gap, chart_width, chart_height),
        )

        values = np.asarray(self.samples, dtype=np.float64)
        for channel, (x, y, width, height) in enumerate(chart_rects):
            self._paint_chart(painter, channel, values, x, y, width, height)

        painter.setPen(QColor("#9aa8b8"))
        painter.drawText(
            margin_left,
            self.height() - 8,
            f"Last {self.WINDOW_SECONDS} seconds",
        )
        painter.end()

    def _paint_chart(
        self,
        painter: QPainter,
        channel: int,
        values: np.ndarray,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        chart = values[:, channel] if values.size else np.empty(0)
        if chart.size:
            low = float(np.min(chart))
            high = float(np.max(chart))
            span = high - low
            padding = max(span * 0.12, abs(high) * 0.03, 1.0)
            low -= padding
            high += padding
        else:
            low, high = -1.0, 1.0

        painter.setPen(QPen(QColor("#293746"), 1))
        for division in range(5):
            grid_y = y + height * division / 4
            painter.drawLine(QPointF(x, grid_y), QPointF(x + width, grid_y))
        painter.drawRect(int(x), int(y), int(width), int(height))

        painter.setPen(QColor("#d9e2ec"))
        painter.drawText(
            8,
            int(y + 18),
            self.channel_labels[channel],
        )
        painter.setPen(QColor("#8795a5"))
        painter.drawText(8, int(y + height - 4), f"{low:.2f}")
        painter.drawText(8, int(y + 32), f"{high:.2f}")
        if not chart.size:
            return

        path = QPainterPath()
        denominator = max(high - low, 1e-12)
        last_index = max(chart.size - 1, 1)
        for index, value in enumerate(chart):
            px = x + width * index / last_index
            py = y + height * (1.0 - (value - low) / denominator)
            if index == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        painter.setPen(QPen(self.COLORS[channel], 1.5))
        painter.drawPath(path)


class SignalViewerWindow(QWidget):
    """Standalone viewer window that can be reopened from the main UI."""

    def __init__(self, sample_rate: int = 250) -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Live EOG Signal — Raw Channels")
        self.resize(760, 510)
        layout = QVBoxLayout(self)
        self.status_label = QLabel("Waiting for serial samples…")
        self.status_label.setStyleSheet("font-size: 14px;")
        self.plot = SignalPlot(sample_rate)
        layout.addWidget(self.status_label)
        layout.addWidget(self.plot, 1)

    def add_samples(self, samples: object) -> None:
        self.plot.add_samples(samples)
        if self.plot.samples:
            first = np.asarray(self.plot.samples[-1], dtype=np.float64)
            self.status_label.setText(
                "Raw Arduino stream · "
                f"CH1 {first[0]:.3f} · CH2 {first[1]:.3f} · "
                f"{len(self.plot.samples)} samples shown"
            )
