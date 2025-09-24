"""Spectral analysis helpers for RTVA."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stft_mag_db(
    frame: NDArray[np.float32] | NDArray[np.float64],
    sr: int,
    n_fft: int | None = None,
    floor_db: float = -120.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return magnitude spectrum in dB for a single frame.

    The result is normalised so that the maximum magnitude in the frame is 0 dB.
    Values below ``floor_db`` are clipped to that floor to stabilise the colour map.
    """

    x = np.asarray(frame, dtype=np.float64)
    if x.size == 0 or sr <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(len(x))))
    n_fft = max(n_fft, len(x))

    spectrum = np.fft.rfft(x, n=n_fft)
    magnitude = np.abs(spectrum)
    if not np.any(magnitude):
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        return freqs, np.full_like(freqs, floor_db, dtype=np.float64)

    mag_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    mag_db -= mag_db.max()
    mag_db = np.maximum(mag_db, floor_db)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return freqs, mag_db
