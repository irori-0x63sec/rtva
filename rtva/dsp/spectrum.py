from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stft_mag_db(
    frames_2d: NDArray[np.floating], sr: int, n_fft: int, hop: int
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Compute the STFT magnitude in dB for a stack of frames."""

    if frames_2d.ndim != 2:
        raise ValueError("frames_2d must be a 2-D array of shape (frames, samples)")

    frames = np.asarray(frames_2d, dtype=np.float64)
    if frames.shape[1] > n_fft:
        n_fft = int(2 ** np.ceil(np.log2(frames.shape[1])))
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = np.abs(spec)
    eps = np.finfo(float).tiny
    S_db = 20.0 * np.log10(np.maximum(mag, eps))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    n_frames = frames.shape[0]
    if n_frames == 0:
        return (
            np.empty((0, 0), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )
    times = np.linspace(-(n_frames - 1) * hop / sr, 0.0, n_frames)

    return S_db.T.astype(np.float32), freqs.astype(np.float32), times.astype(np.float32)


def h1_h2_db(frame: NDArray[np.floating], sr: int, f0: float) -> float:
    """Placeholder for future H1-H2 implementation (not yet in MVP)."""

    raise NotImplementedError
