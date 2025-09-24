from __future__ import annotations

import numpy as np

from rtva.dsp.spectrum import stft_mag_db


def test_stft_mag_db_shapes_and_axes():
    sr = 16000
    duration = 0.04
    hop = int(0.01 * sr)
    frame_len = int(0.032 * sr)
    win = np.hanning(frame_len)
    t = np.arange(0, duration, 1 / sr)
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    S_db, freqs, times = stft_mag_db(signal, sr, n_fft=1024, hop=hop, win=win)

    assert S_db.shape[0] == 513  # n_fft // 2 + 1
    assert freqs.shape[0] == S_db.shape[0]
    assert times.shape[0] == S_db.shape[1]
    np.testing.assert_allclose(times[-1], 0.0)
    expected_frames = 1 + (len(signal) - frame_len) // hop
    np.testing.assert_allclose(times.shape[0], expected_frames)

    peak_idx = np.argmax(S_db[:, -1])
    peak_freq = freqs[peak_idx]
    assert abs(peak_freq - 440) < 30
