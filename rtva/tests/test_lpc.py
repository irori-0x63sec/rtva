import numpy as np
import pytest
from scipy.signal import lfilter

from rtva.dsp.lpc import lpc_formants


def synth_vowel(formants, bandwidths, sr, duration):
    rng = np.random.default_rng(0)
    n_samples = int(sr * duration)
    src = rng.normal(0, 1, n_samples)
    y = src
    for f, bw in zip(formants, bandwidths):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        a = np.array([1, -2 * r * np.cos(theta), r**2])
        b = np.array([1 - r])
        y = lfilter(b, a, y)
    return y.astype(np.float32)


def test_lpc_formants_tracks_resonances():
    sr = 16000
    formants = [500, 1500, 2500]
    bandwidths = [60, 90, 150]
    frame = synth_vowel(formants, bandwidths, sr, duration=0.05)

    f1, f2, f3 = lpc_formants(frame * np.hanning(len(frame)), sr, order=14, fmax=4000)

    assert f1 == pytest.approx(formants[0], rel=0.2)
    assert f2 == pytest.approx(formants[1], rel=0.2)
    assert f3 == pytest.approx(formants[2], rel=0.2)
