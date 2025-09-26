from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from rtva.dsp.lpc import burg_formants_parselmouth, lpc_formants

try:
    import parselmouth  # noqa: F401

    PARSELMOUTH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PARSELMOUTH_AVAILABLE = False


def _synth_vowel(formants: tuple[float, float, float], sr: int, duration: float) -> np.ndarray:
    rng = np.random.default_rng(1234)
    excitation = rng.standard_normal(int(sr * duration)).astype(np.float32)
    signal = excitation
    bandwidths = (80.0, 90.0, 160.0)
    for f, bw in zip(formants, bandwidths):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        a = np.array([1.0, -2.0 * r * np.cos(theta), r**2], dtype=np.float64)
        b = np.array([1.0 - r], dtype=np.float64)
        signal = lfilter(b, a, signal)
    return signal.astype(np.float32)


@pytest.mark.parametrize("method", ["lpc", "burg"])
def test_formant_estimators_reasonable(method: str) -> None:
    sr = 16000
    targets = (700.0, 1200.0, 2500.0)
    tone = _synth_vowel(targets, sr=sr, duration=0.2)
    start = int(0.08 * sr)
    frame = tone[start : start + int(0.05 * sr)]

    if method == "burg":
        if not PARSELMOUTH_AVAILABLE:
            pytest.xfail("parselmouth not available")
        f1, f2, f3 = burg_formants_parselmouth(frame, sr, fmax=4000)[:3]
        if (f1, f2, f3) == (0.0, 0.0, 0.0):
            pytest.xfail("parselmouth backend unavailable")
    else:
        f1, f2, f3 = lpc_formants(frame, sr, order=16, fmax=4000)[:3]

    estimates = (f1, f2, f3)
    for est, target in zip(estimates[:2], targets[:2]):
        assert target * 0.85 < est < target * 1.15


def test_lpc_returns_zero_for_silence() -> None:
    sr = 16000
    silence = np.zeros(int(0.04 * sr), dtype=np.float32)
    f1, f2, f3 = lpc_formants(silence, sr)[:3]
    assert (f1, f2, f3) == (0.0, 0.0, 0.0)
