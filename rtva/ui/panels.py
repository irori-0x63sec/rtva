from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class PitchPanel(QWidget):
    """F0数値＋直近時系列（10秒くらい）"""

    def __init__(self, timespan_sec: int = 10, sr_hop: int = 100) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        head = QHBoxLayout()
        self.value_label = QLabel("F0: -- Hz")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.stability_label = QLabel("±cent: --")
        head.addWidget(self.value_label)
        head.addWidget(self.stability_label)
        layout.addLayout(head)

        self.plot = pg.PlotWidget()
        self.plot.setYRange(60, 600)  # 目安
        self.plot.setLabel("left", "F0 (Hz)")
        self.plot.setLabel("bottom", "time (s)")
        self.curve = self.plot.plot(pen=pg.mkPen(width=2))
        layout.addWidget(self.plot)

        self.capacity = int(timespan_sec * sr_hop)
        self.buf = np.full(self.capacity, np.nan, dtype=float)
        self.index = 0

    def update_pitch(self, f0_hz: float) -> None:
        # リングバッファに追記
        self.buf[self.index % self.capacity] = f0_hz if f0_hz > 0 else np.nan
        self.index += 1

        # 円環→直列に展開して描画
        y = self.buf.take(np.arange(self.capacity), mode="wrap")
        # 100Hz更新想定で時間軸を生成（容量に合わせて -T..0 の等間隔）
        if len(y) > 0:
            x = np.linspace(-len(y) / 100.0, 0.0, len(y))
            self.curve.setData(x, y)

        # ラベル更新
        if f0_hz > 0:
            self.value_label.setText(f"F0: {f0_hz:6.1f} Hz")
        else:
            self.value_label.setText("F0: -- Hz")

    def update_stability(self, cents_std: float | None) -> None:
        if cents_std is None or (isinstance(cents_std, float) and np.isnan(cents_std)):
            self.stability_label.setText("±cent: --")
        else:
            self.stability_label.setText(f"±cent: {cents_std:4.1f}")


class SpectroPanel(QWidget):
    """スペクトログラムとフォームントのオーバーレイ表示。"""

    def __init__(self, fmax: int = 6000) -> None:
        super().__init__()
        self.fmax = fmax
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Frequency (Hz)")
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setYRange(0, fmax)
        self.plot.setLimits(yMin=0, yMax=fmax)

        cmap = pg.colormap.get("magma")
        self.image_item = pg.ImageItem()
        self.image_item.setLookupTable(cmap.getLookupTable(alpha=False))
        self.image_item.setLevels([-120.0, 0.0])
        self.plot.addItem(self.image_item)

        pens = [
            pg.mkPen("#ff6f61", width=2),
            pg.mkPen("#6abf69", width=2),
            pg.mkPen("#5b8def", width=2),
        ]
        self.formant_curves = [self.plot.plot(pen=pen) for pen in pens]

        layout.addWidget(self.plot)

    def update(
        self,
        S_db: np.ndarray,
        freqs: np.ndarray,
        times: np.ndarray,
        formants: list[np.ndarray] | None,
    ) -> None:
        if S_db.size == 0 or freqs.size == 0 or times.size == 0:
            return

        self.image_item.resetTransform()
        self.image_item.setImage(S_db, autoLevels=False)

        if times.size > 1:
            dt = float(times[1] - times[0])
        else:
            dt = 1.0
        if freqs.size > 1:
            df = float(freqs[1] - freqs[0])
        else:
            df = float(freqs[-1]) if freqs.size else 1.0

        self.image_item.scale(dt, -df)
        self.image_item.setPos(float(times[0]), float(freqs[-1]))

        self.plot.setXRange(float(times[0]), float(times[-1]), padding=0.0)
        ymax = min(float(freqs[-1]), float(self.fmax))
        self.plot.setYRange(0.0, ymax, padding=0.0)
        self.plot.setLimits(xMin=float(times[0]), xMax=0.0, yMin=0.0, yMax=float(freqs[-1]))

        if formants is None:
            for curve in self.formant_curves:
                curve.setData([], [])
            return

        for idx, curve in enumerate(self.formant_curves):
            if idx < len(formants) and len(formants[idx]) == len(times):
                values = np.asarray(formants[idx], dtype=float)
                values = np.clip(values, 0.0, float(freqs[-1]))
                curve.setData(times, values)
            else:
                curve.setData([], [])


class HarmonicsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("H1–H2 (coming soon)"))


class CPPPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("CPP (coming soon)"))
