import numpy as np
from scipy.signal import lfilter

from rtva.dsp.lpc import burg_formants_parselmouth, lpc_formants


def _synth_vowel(
    sr: int,
    duration: float,
    formants: tuple[float, float, float],
    bandwidths: tuple[float, float, float],
    f0: float = 120.0,
) -> np.ndarray:
    n = int(duration * sr)
    source = np.zeros(n, dtype=np.float64)
    period = max(1, int(sr / f0))
    source[::period] = 1.0

    signal = source
    for freq, bw in zip(formants, bandwidths):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * freq / sr
        a = np.array([1.0, -2.0 * r * np.cos(theta), r * r], dtype=np.float64)
        b = np.array([1.0 - r], dtype=np.float64)
        signal = lfilter(b, a, signal)

    return signal.astype(np.float32)


def _test_frame() -> tuple[np.ndarray, int]:
    sr = 16000
    frame_len = int(0.04 * sr)
    waveform = _synth_vowel(sr, 0.4, (500.0, 1500.0, 2500.0), (80.0, 90.0, 120.0))
    start = waveform.size // 2 - frame_len // 2
    frame = waveform[start : start + frame_len]
    return frame, sr


def _assert_formants_close(
    observed: tuple[float, float, float], target: tuple[float, float, float]
):
    for value, expect in zip(observed, target):
        if expect <= 0:
            continue
        assert expect * 0.85 <= value <= expect * 1.15


def test_lpc_formants_synthetic_vowel():
    frame, sr = _test_frame()
    f1, f2, f3 = lpc_formants(frame, sr, order=14, fmax=4000)
    _assert_formants_close((f1, f2, f3), (500.0, 1500.0, 2500.0))


def test_burg_formants_synthetic_vowel():
    frame, sr = _test_frame()
    f1, f2, f3 = burg_formants_parselmouth(frame, sr, fmax=4000)
    _assert_formants_close((f1, f2, f3), (500.0, 1500.0, 2500.0))
