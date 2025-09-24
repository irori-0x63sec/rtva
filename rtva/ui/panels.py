from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class PitchPanel(QWidget):
    """F0数値＋直近時系列（10秒くらい）"""

    def __init__(self, timespan_sec=10, sr_hop=100):
        super().__init__()
        self.setLayout(QVBoxLayout())
        head = QHBoxLayout()
        self.value_label = QLabel("F0: -- Hz")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.stability_label = QLabel("±cent: --")
        head.addWidget(self.value_label)
        head.addWidget(self.stability_label)
        self.layout().addLayout(head)

        self.plot = pg.PlotWidget()
        self.plot.setYRange(60, 600)  # 目安
        self.plot.setLabel("left", "F0 (Hz)")
        self.plot.setLabel("bottom", "time (s)")
        self.curve = self.plot.plot(pen=pg.mkPen(width=2))
        self.layout().addWidget(self.plot)

        self.capacity = int(timespan_sec * sr_hop)
        self.buf = np.full(self.capacity, np.nan, dtype=float)
        self.index = 0

    def update_pitch(self, f0_hz: float):
        self.buf[self.index % self.capacity] = f0_hz if f0_hz > 0 else np.nan
        self.index += 1
        # 円環→直列へ
        start = self.index - self.capacity
        idx = np.arange(start, self.index)
        y = self.buf.take(np.arange(self.capacity), mode="wrap")
        x = np.linspace(-len(y) / 100, 0, len(y))  # 100Hz更新想定
        self.curve.setData(x, y)
        if f0_hz > 0:
            self.value_label.setText(f"F0: {f0_hz:6.1f} Hz")
        else:
            self.value_label.setText("F0: -- Hz")

    def update_stability(self, cents_std: float | None):
        if cents_std is None or np.isnan(cents_std):
            self.stability_label.setText("±cent: --")
        else:
            self.stability_label.setText(f"±cent: {cents_std:4.1f}")


class SpectroPanel(QWidget):
    """Spectrogram heatmap with formant overlays."""

    def __init__(self, timespan_sec: float = 6.0, hop_s: float = 0.01, fmax: float = 8000.0):
        super().__init__()
        self.setLayout(QVBoxLayout())

        self.timespan_sec = float(timespan_sec)
        self.hop_s = max(float(hop_s), 1e-3)
        self.fmax = float(fmax)
        self.time_bins = max(4, int(np.ceil(self.timespan_sec / self.hop_s)))
        self.freqs: np.ndarray | None = None
        self.spec_buf: np.ndarray | None = None
        self.formant_hist: np.ndarray | None = None
        self.time_axis = np.linspace(-self.timespan_sec, 0.0, self.time_bins)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setYRange(0, self.fmax)
        self.plot.setXRange(-self.timespan_sec, 0.0)
        self.plot.showGrid(x=False, y=True, alpha=0.2)

        self.image = pg.ImageItem()
        cmap = pg.colormap.get("magma")
        self.image.setLookupTable(cmap.getLookupTable())
        self.plot.addItem(self.image)

        colors = ["#66c2a5", "#fc8d62", "#8da0cb"]
        self.formant_curves = [
            self.plot.plot(pen=pg.mkPen(color=colors[i], width=2)) for i in range(3)
        ]

        self.layout().addWidget(self.plot)

    def _ensure_buffers(self, freqs: np.ndarray):
        mask = freqs <= self.fmax
        sel = freqs[mask]
        if sel.size == 0:
            return
        if self.freqs is None or self.freqs.size != sel.size:
            self.freqs = sel
            self.spec_buf = np.full((sel.size, self.time_bins), -120.0, dtype=np.float32)
            self.formant_hist = np.full((3, self.time_bins), np.nan, dtype=np.float32)
            self.image.setImage(np.flipud(self.spec_buf), autoLevels=False)
            self.image.setRect(QRectF(-self.timespan_sec, 0.0, self.timespan_sec, self.freqs[-1]))
            self.plot.setYRange(0, min(self.fmax, float(self.freqs[-1])))

    def update_spectrogram(self, freqs: np.ndarray, mag_db: np.ndarray):
        if freqs.size == 0 or mag_db.size == 0:
            return
        self._ensure_buffers(freqs)
        if self.freqs is None or self.spec_buf is None:
            return
        mask = freqs <= self.freqs[-1]
        sel = mag_db[mask][: self.freqs.size]
        if sel.size != self.freqs.size:
            return
        self.spec_buf[:, :-1] = self.spec_buf[:, 1:]
        self.spec_buf[:, -1] = sel
        self.image.setImage(np.flipud(self.spec_buf), autoLevels=False)

    def update_formants(self, f1: float, f2: float, f3: float):
        if self.formant_hist is None:
            self.formant_hist = np.full((3, self.time_bins), np.nan, dtype=np.float32)
        self.formant_hist[:, :-1] = self.formant_hist[:, 1:]
        self.formant_hist[:, -1] = [
            f1 if f1 > 0 else np.nan,
            f2 if f2 > 0 else np.nan,
            f3 if f3 > 0 else np.nan,
        ]
        for idx, curve in enumerate(self.formant_curves):
            curve.setData(self.time_axis, self.formant_hist[idx])


class HarmonicsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("H1–H2 (coming soon)"))


class CPPPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("CPP (coming soon)"))
