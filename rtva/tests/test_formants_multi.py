from __future__ import annotations

import numpy as np
from scipy import signal

from rtva.dsp.lpc import lpc_formants


def test_lpc_formants_returns_five_reasonable_values() -> None:
    sr = 16000
    duration = 0.3
    rng = np.random.default_rng(1234)
    noise = rng.standard_normal(int(sr * duration)).astype(np.float32)

    target_formants = [500.0, 1500.0, 2500.0, 3300.0, 4200.0]
    bandwidths = [60.0, 80.0, 120.0, 160.0, 200.0]

    a = np.array([1.0])
    for f, bw in zip(target_formants, bandwidths):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        pole = np.array([1.0, -2 * r * np.cos(theta), r**2])
        a = np.convolve(a, pole)

    vowel = signal.lfilter([1.0], a, noise)
    frame = vowel[int(0.1 * sr) : int(0.1 * sr) + 1024]

    estimates = lpc_formants(frame.astype(np.float32), sr, order=20, fmax=5000, n_formants=5)
    assert len(estimates) == 5
    assert all(np.isfinite(est) for est in estimates)

    for expected, got in zip(target_formants[:3], estimates[:3]):
        assert abs(got - expected) / expected < 0.4

    for got in estimates[3:]:
        assert got > 0
