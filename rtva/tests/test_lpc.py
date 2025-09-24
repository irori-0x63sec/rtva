from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from rtva.dsp.lpc import lpc_formants


def _synth_vowel(
    formants: list[float], bandwidths: list[float], sr: int, duration: float
) -> np.ndarray:
    rng = np.random.default_rng(0)
    excitation = rng.standard_normal(int(sr * duration)).astype(np.float32)
    signal = excitation
    for f, bw in zip(formants, bandwidths):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        a = np.array([1.0, -2.0 * r * np.cos(theta), r**2])
        b = np.array([1.0 - r])
        signal = lfilter(b, a, signal)
    return signal.astype(np.float32)


def test_lpc_formants_tracks_targets():
    sr = 16000
    target_formants = [750.0, 1100.0, 2100.0]
    bandwidths = [80.0, 100.0, 160.0]
    tone = _synth_vowel(target_formants, bandwidths, sr=sr, duration=0.2)
    start = int(0.05 * sr)
    frame = tone[start : start + int(0.04 * sr)]
    f1, f2, f3 = lpc_formants(frame, sr, order=14, fmax=4000)

    results = [f1, f2, f3]
    for est, target in zip(results, target_formants):
        assert target * 0.85 < est < target * 1.15


def test_lpc_formants_silence_returns_zero():
    sr = 16000
    silence = np.zeros(int(0.04 * sr), dtype=np.float32)
    f1, f2, f3 = lpc_formants(silence, sr, order=12)
    assert f1 == 0.0 and f2 == 0.0 and f3 == 0.0
