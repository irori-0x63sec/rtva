from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
import numpy as np
from rtva.ui.panels import PitchPanel, SpectroPanel, HarmonicsPanel, CPPPanel
from rtva.audio import AudioStream, hann
from rtva.dsp.pitch import yin_f0

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTVA — Real-Time Voice Analyzer")

        # === GUI ===
        central = QWidget(self)
        lay = QVBoxLayout(central)
        self.pitch = PitchPanel(timespan_sec=10, sr_hop=100)
        self.spectro = SpectroPanel()
        self.harm = HarmonicsPanel()
        self.cpp = CPPPanel()
        lay.addWidget(self.pitch)
        lay.addWidget(self.spectro)
        lay.addWidget(self.harm)
        lay.addWidget(self.cpp)
        self.setCentralWidget(central)

        # === Audio / DSP ===
        self.sr = 44100
        self.hop_ms = 10         # 10msごと更新 ≒ 100 Hz
        self.win_ms = 32
        self.blocksize = int(self.sr * self.hop_ms / 1000)
        self.win = int(self.sr * self.win_ms / 1000)
        self.win_buf = np.zeros(self.win, dtype=np.float32)
        self.win_hann = hann(self.win)

        self.stream = AudioStream(sr=self.sr, blocksize=self.blocksize)
        self.stream.start()

        # ~100Hz でループ
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.hop_ms)

    def closeEvent(self, e):
        try:
            self.stream.stop()
        except Exception:
            pass
        return super().closeEvent(e)

    def _tick(self):
        # 新ブロックを取り出し、スライディングウィンドウ更新
        try:
            hop = self.stream.read()
        except Exception:
            return
        self.win_buf = np.roll(self.win_buf, -len(hop))
        self.win_buf[-len(hop):] = hop
        frame = (self.win_buf * self.win_hann).astype(np.float32)

        # F0推定（YIN）
        f0 = yin_f0(frame, self.sr, fmin=75, fmax=350)  # 後でUIから変更可
        self.pitch.update_pitch(f0)

        # 簡易：直近0.5秒の±centゆらぎ
        cents_std = self._cents_std_recent()
        self.pitch.update_stability(cents_std)

    def _cents_std_recent(self, n=50):
        # PitchPanelのリングバッファから直近n点の標準偏差[cent]
        buf = self.pitch.buf.copy()
        # nanを除外
        vals = buf[~np.isnan(buf)]
        if len(vals) < max(5, n//2):
            return None
        recent = vals[-n:]
        hz = recent[recent > 0]
        if len(hz) < 5:
            return None
        # 中央値からのcent変換
        med = np.median(hz)
        cents = 1200 * np.log2(hz / med)
        return float(np.std(cents))
