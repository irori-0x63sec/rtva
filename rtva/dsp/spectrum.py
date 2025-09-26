from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stft_mag_db(
    x: NDArray[np.floating], sr: int, n_fft: int, hop: int, win: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Compute the STFT magnitude in dB for a 1-D signal."""

    sig = np.asarray(x, dtype=np.float32).reshape(-1)
    if sig.size == 0:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    window = np.asarray(win, dtype=np.float32).reshape(-1)
    if window.size == 0:
        raise ValueError("win must be a non-empty 1-D array")

    frame_len = window.size
    if hop <= 0:
        raise ValueError("hop must be positive")

    if sig.size < frame_len:
        pad = frame_len - sig.size
        sig = np.pad(sig, (pad, 0))

    if sig.size > frame_len:
        remainder = (sig.size - frame_len) % hop
        if remainder:
            sig = sig[remainder:]

    if n_fft < frame_len:
        n_fft = 1 << int(np.ceil(np.log2(frame_len)))

    frames = np.lib.stride_tricks.sliding_window_view(sig, frame_len)[::hop]
    if frames.size == 0:
        frames = sig[-frame_len:].reshape(1, -1)

    frames = frames.astype(np.float32) * window
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = np.abs(spec)
    S_db = 20.0 * np.log10(np.maximum(mag, 1e-8))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)
    n_frames = frames.shape[0]
    times = np.linspace(-(n_frames - 1) * hop / sr, 0.0, n_frames).astype(np.float32)

    return S_db.T.astype(np.float32, copy=False), freqs, times


def h1_h2_db(frame: NDArray[np.floating], sr: int, f0: float) -> float:
    """Estimate the H1-H2 spectral level difference in dB."""

    if f0 <= 0 or not np.isfinite(f0):
        return float(np.nan)

    x = np.asarray(frame, dtype=np.float64)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return float(np.nan)

    x = x * np.hanning(x.size)
    if not np.any(np.abs(x) > 0):
        return float(np.nan)

    n_fft = max(2048, 1 << int(np.ceil(np.log2(x.size * 2))))
    spec = np.fft.rfft(x, n=n_fft)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    def _peak(target: float) -> float | None:
        if target >= freqs[-1]:
            return None
        bw = max(20.0, target * 0.25)
        mask = np.logical_and(freqs >= target - bw, freqs <= target + bw)
        if not np.any(mask):
            return None
        return float(np.max(mag[mask]))

    h1 = _peak(f0)
    h2 = _peak(2.0 * f0)
    if h1 is None or h2 is None or h1 <= 0 or h2 <= 0:
        return float(np.nan)

    return 20.0 * np.log10((h1 + 1e-12) / (h2 + 1e-12))


def hnr_db(frame: NDArray[np.floating], sr: int, f0: float) -> float:
    """Estimate the harmonic-to-noise ratio in dB."""

    if f0 <= 0 or not np.isfinite(f0):
        return float(np.nan)

    x = np.asarray(frame, dtype=np.float64)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return float(np.nan)

    x = x - np.mean(x)
    if not np.any(np.abs(x) > 0):
        return float(np.nan)

    autocorr = np.correlate(x, x, mode="full")
    autocorr = autocorr[x.size - 1 :]
    if autocorr.size == 0 or autocorr[0] <= 0:
        return float(np.nan)

    lag = int(round(sr / float(f0)))
    if lag <= 0 or lag >= autocorr.size:
        return float(np.nan)

    r0 = autocorr[0]
    r_tau = autocorr[lag]
    if r_tau <= 0:
        return float(np.nan)

    noise = max(r0 - r_tau, 1e-12)
    ratio = r_tau / noise
    return 10.0 * np.log10(max(ratio, 1e-12))
