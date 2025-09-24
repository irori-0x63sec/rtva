import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
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
    """後でスペクトログラム/フォルマントを載せる土台（今はプレースホルダ）"""

    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Spectrogram (coming soon)"))


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
