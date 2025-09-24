from __future__ import annotations

import numpy as np


def stft_mag_db(
    frames_2d: np.ndarray, sr: int, n_fft: int, hop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log magnitude spectrogram (dB) with frequency/time axes."""

    if frames_2d.ndim != 2:
        raise ValueError("frames_2d must be 2D (n_frames, frame_len)")

    n_frames, frame_len = frames_2d.shape
    if n_frames == 0:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    fft_size = max(n_fft, frame_len)
    spectrum = np.fft.rfft(frames_2d, n=fft_size, axis=1)
    magnitude = np.abs(spectrum)
    magnitude = np.maximum(magnitude, 1e-10)
    db = 20 * np.log10(magnitude)
    db -= np.max(db)
    db = np.clip(db, -80.0, 0.0)

    freq_axis = np.linspace(0.0, sr / 2, db.shape[1], dtype=np.float32)
    duration = (n_frames - 1) * hop / sr
    time_axis = np.linspace(-duration, 0.0, n_frames, dtype=np.float32)

    return db.T.astype(np.float32), freq_axis, time_axis
