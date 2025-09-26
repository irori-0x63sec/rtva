from __future__ import annotations

import numpy as np

from rtva.dsp.cepstrum import cpp_db
from rtva.dsp.spectrum import h1_h2_db, hnr_db


def test_h1h2_positive_for_strong_fundamental() -> None:
    sr = 16000
    t = np.arange(0, int(0.05 * sr)) / sr
    fundamental = np.sin(2 * np.pi * 200 * t)
    second = 0.3 * np.sin(2 * np.pi * 400 * t)
    frame = (fundamental + second).astype(np.float32)

    value = h1_h2_db(frame, sr, f0=200.0)
    assert np.isfinite(value)
    assert value > 0.0


def test_hnr_decreases_with_noise() -> None:
    sr = 22050
    t = np.arange(0, int(0.04 * sr)) / sr
    voiced = np.sin(2 * np.pi * 150 * t).astype(np.float32)
    noisy = voiced + 0.5 * np.random.default_rng(1).standard_normal(voiced.size)

    hnr_clean = hnr_db(voiced, sr, f0=150.0)
    hnr_noisy = hnr_db(noisy.astype(np.float32), sr, f0=150.0)

    assert np.isfinite(hnr_clean)
    assert hnr_clean > hnr_noisy


def test_cpp_decreases_with_noise() -> None:
    sr = 16000
    t = np.arange(0, int(0.06 * sr)) / sr
    clean = np.sin(2 * np.pi * 180 * t).astype(np.float32)
    rng = np.random.default_rng(42)
    noisy = (clean + 2.0 * rng.standard_normal(clean.size)).astype(np.float32)

    cpp_clean = cpp_db(clean, sr)
    cpp_noisy = cpp_db(noisy, sr)

    assert np.isfinite(cpp_clean)
    assert cpp_clean > cpp_noisy
