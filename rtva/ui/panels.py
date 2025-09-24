from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
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
        # よく見る帯に固定（必要なら後で自動化）
        self.plot.setYRange(80, 500)
        self.plot.setLabel("left", "F0 (Hz)")
        self.plot.setLabel("bottom", "time (s)")
        self.curve = self.plot.plot(pen=pg.mkPen(width=2))
        layout.addWidget(self.plot)

        self.capacity = int(timespan_sec * sr_hop)
        self.buf = np.full(self.capacity, np.nan, dtype=float)
        self.index = 0

    def update_pitch(self, f0_hz: float) -> None:
        self.buf[self.index % self.capacity] = f0_hz if f0_hz > 0 else np.nan
        self.index += 1

        y = self.buf.take(np.arange(self.capacity), mode="wrap")
        if len(y) > 0:
            x = np.linspace(-len(y) / 100.0, 0.0, len(y))  # 100Hz更新想定
            self.curve.setData(x, y)

        self.value_label.setText(f"F0: {f0_hz:6.1f} Hz" if f0_hz > 0 else "F0: -- Hz")

    def update_stability(self, cents_std: float | None) -> None:
        if cents_std is None or (isinstance(cents_std, float) and np.isnan(cents_std)):
            self.stability_label.setText("±cent: --")
        else:
            self.stability_label.setText(f"±cent: {cents_std:4.1f}")


class SpectroPanel(QWidget):
    """
    スペクトログラム + フォルマントオーバーレイ
    ・実座標は setRect で指定
    ・dBレンジはパーセンタイル＋EMAで自動追従
    ・表示は直近 window_sec のみ
    ・周波数は 0..fmax_hz に制限
    """

    def __init__(self, window_sec: float = 1.0, fmax_hz: float = 6000.0):
        super().__init__()
        self.setLayout(QVBoxLayout())

        self.plot = pg.PlotWidget()
        self.plot.invertY(True)  # 周波数が上へ
        self.plot.setLabel("left", "Frequency (Hz)")
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setYRange(0, fmax_hz)
        self.plot.setLimits(yMin=0, yMax=fmax_hz)
        self.fmax_hz = float(fmax_hz)

        # 画像
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)

        # LUT（見やすいカラーマップが欲しければ 'magma' 等に変更可）
        # lut = pg.colormap.get('magma').getLookupTable(alpha=True)
        # self.image_item.setLookupTable(lut)

        # dB レンジ（EMA初期値）
        self._lv_ema = (-80.0, 0.0)
        self.image_item.setLevels(self._lv_ema)

        # フォルマント
        pen = pg.mkPen(width=2)
        self.f1_curve = self.plot.plot(pen=pen)
        self.f2_curve = self.plot.plot(pen=pen)
        self.f3_curve = self.plot.plot(pen=pen)

        self.layout().addWidget(self.plot)
        self.window_sec = float(window_sec)

    def _update_levels(self, S_db: np.ndarray, alpha: float = 0.2) -> tuple[float, float]:
        # ノイズ耐性を持った自動スケール（5〜98パーセンタイル）
        vmin = float(np.nanpercentile(S_db, 5))
        vmax = float(np.nanpercentile(S_db, 98))
        if vmax - vmin < 10.0:
            vmin, vmax = vmin - 5.0, vmin + 5.0
        # EMAでスムージング
        lv0, lv1 = self._lv_ema
        nv0 = (1 - alpha) * lv0 + alpha * vmin
        nv1 = (1 - alpha) * lv1 + alpha * vmax
        self._lv_ema = (nv0, nv1)
        return self._lv_ema

    def update(
        self,
        S_db: np.ndarray,
        freqs: np.ndarray,
        times: np.ndarray,
        formants: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ):
        if S_db is None or S_db.size == 0:
            return

        # numpyの型・メモリ整理
        S_db = np.asarray(S_db, dtype=np.float32, order="C")
        freqs = np.asarray(freqs, dtype=np.float32)
        times = np.asarray(times, dtype=np.float32)

        # 表示は直近 window_sec のみ
        if len(times) > 1:
            t_end = float(times[-1])
            t_start = t_end - self.window_sec
            mask_t = times >= t_start
            times_v = times[mask_t]
            S_db = S_db[:, mask_t]
        else:
            times_v = times

        # 周波数は 0..fmax_hz へ制限
        mask_f = freqs <= self.fmax_hz
        freqs_v = freqs[mask_f]
        S_db = S_db[mask_f, :]

        if S_db.size == 0 or len(freqs_v) == 0 or len(times_v) == 0:
            return

        # dBレンジを更新
        levels = self._update_levels(S_db)
        try:
            self.image_item.setImage(S_db, autoLevels=False, levels=levels)
        except Exception as e:
            if not hasattr(self, "_img_err_once"):
                print(f"[WARN] spectrogram render: {e}")
                self._img_err_once = True
            return

        # 実座標の矩形（時間×周波数）
        t0 = float(times_v[0])
        t1 = float(times_v[-1]) if len(times_v) > 1 else (t0 + 1e-3)
        f0 = float(freqs_v[0])
        f1 = float(freqs_v[-1]) if len(freqs_v) > 1 else (f0 + 1.0)
        self.image_item.resetTransform()
        self.image_item.setRect(QRectF(t0, f0, (t1 - t0), (f1 - f0)))

        # フォルマントを同じ時間窓で描画
        if formants is not None:
            F1, F2, F3 = formants
            if len(F1) == len(times):
                F1 = F1[mask_t]
                F2 = F2[mask_t]
                F3 = F3[mask_t]
            self.f1_curve.setData(times_v, F1)
            self.f2_curve.setData(times_v, F2)
            self.f3_curve.setData(times_v, F3)


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
