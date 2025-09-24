from queue import Queue

import numpy as np
import sounddevice as sd


class AudioStream:
    """Mic → ライブフレーム供給。hopごとに非同期で取り出せる"""

    def __init__(self, sr=44100, blocksize=1024):
        self.sr = sr
        self.blocksize = blocksize
        self.q = Queue(maxsize=64)
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sr,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time, status):
        if status:
            # XRunsなどのステータス。必要ならログへ。
            pass
        try:
            self.q.put_nowait(indata[:, 0].copy())
        except Exception:  # ★ bare except を明示的に
            # バックプレッシャ時は捨てる
            pass

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

    def read(self):
        """ブロックを1つ返す（float32, shape=(blocksize,)）"""
        return self.q.get()


def hann(n: int):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
