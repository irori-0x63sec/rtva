from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cpp_db(frame: NDArray[np.floating], sr: int, frame_ms: int = 46) -> float:
    """Compute the cepstral peak prominence in dB."""

    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")

    n_win = int(round(sr * frame_ms / 1000))
    if n_win <= 0:
        return float(np.nan)

    x = np.asarray(frame, dtype=np.float64)
    if x.size == 0 or not np.any(np.isfinite(x)):
        return float(np.nan)

    if x.size < n_win:
        pad = n_win - x.size
        x = np.pad(x, (0, pad))
    else:
        x = x[:n_win]

    x = x * np.hanning(n_win)
    if not np.any(np.abs(x) > 0):
        return float(np.nan)

    n_fft = 1 << int(np.ceil(np.log2(n_win * 2)))
    spec = np.fft.rfft(x, n=n_fft)
    mag = np.abs(spec) + 1e-12
    log_mag = 20.0 * np.log10(mag)
    cep = np.fft.irfft(log_mag, n=n_fft)

    quef = np.arange(n_fft) / float(sr)
    q_min = 1.0 / 500.0  # ≈ 500 Hz
    q_max = 1.0 / 60.0  # ≈ 60 Hz
    mask = (quef >= q_min) & (quef <= q_max)
    if not np.any(mask):
        return float(np.nan)

    q_sel = quef[mask]
    cep_sel = cep[mask]
    peak_idx = int(np.argmax(cep_sel))
    peak_val = float(cep_sel[peak_idx])
    q_peak = float(q_sel[peak_idx])

    if q_sel.size < 2:
        return float(np.nan)

    baseline_interp = float(np.interp(q_peak, [q_sel[0], q_sel[-1]], [cep_sel[0], cep_sel[-1]]))
    return float(max(peak_val - baseline_interp, 0.0))
