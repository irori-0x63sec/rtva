from __future__ import annotations

import numpy as np

from rtva.dsp.vad import rms_db, simple_vad


def test_rms_db_matches_expected_value() -> None:
    frame = np.full(480, 0.5, dtype=np.float32)
    level = rms_db(frame)
    assert np.isfinite(level)
    assert abs(level + 6.0206) < 0.5  # 0.5 amplitude ≈ -6 dB


def test_simple_vad_detects_tone() -> None:
    sr = 48000
    t = np.arange(0, int(0.03 * sr)) / sr
    frame = 0.1 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    assert simple_vad(frame, sr, thr_db=-40.0)


def test_simple_vad_rejects_quiet_noise() -> None:
    sr = 48000
    rng = np.random.default_rng(123)
    noise = (1e-3 * rng.standard_normal(int(0.03 * sr))).astype(np.float32)
    assert not simple_vad(noise, sr, thr_db=-40.0)
