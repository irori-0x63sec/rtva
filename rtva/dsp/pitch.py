# rtva/dsp/pitch.py
# librosa/numba 依存を避けるためのピッチ推定。
# 優先: Praat (parselmouth) / 失敗時: 自前オートコリレーションにフォールバック。

from __future__ import annotations

import numpy as np

try:
    import parselmouth  # Praat bindings
except Exception:  # parselmouthが無い・動かない場合も考慮
    parselmouth = None


def _pitch_parselmouth(
    frame: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 350.0
) -> float:
    """Praat(autocorrelation)でF0を推定。中央値でロバスト化。"""
    snd = parselmouth.Sound(frame.astype(float), sampling_frequency=sr)
    # time_step=0 は自動。autocorrelation method が既定。
    pitch = snd.to_pitch(time_step=0.0, pitch_floor=fmin, pitch_ceiling=fmax)
    values = pitch.selected_array["frequency"]  # Hz, 無声は 0
    vals = values[values > 0]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def _pitch_autocorr(frame: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 350.0) -> float:
    """軽量なオートコリレーションによるF0推定（簡易版）。"""
    x = frame.astype(np.float64)
    x -= x.mean()
    if np.allclose(x, 0.0):
        return 0.0

    n = len(x)
    # FFTベースの自己相関（高速）
    nfft = 1 << ((n - 1).bit_length() * 2)
    X = np.fft.rfft(x, nfft)
    r = np.fft.irfft(X * np.conj(X))[:n]  # 正ラグのみ

    # 周期の探索レンジ
    max_period = int(sr / fmin)
    min_period = int(sr / fmax)
    min_period = max(2, min_period)
    if max_period >= len(r):
        max_period = len(r) - 1

    seg = r[min_period:max_period]
    if seg.size <= 0:
        return 0.0
    lag = int(np.argmax(seg)) + min_period
    if lag <= 0:
        return 0.0
    return float(sr / lag)


def yin_f0(frame: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 350.0) -> float:
    """以前のAPI名(yin_f0)をそのままにして互換維持。"""
    if parselmouth is not None:
        try:
            return _pitch_parselmouth(frame, sr, fmin, fmax)
        except Exception:
            pass
    return _pitch_autocorr(frame, sr, fmin, fmax)
