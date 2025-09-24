from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rms_db(frame: NDArray[np.floating]) -> float:
    """Return RMS in decibels."""

    x = np.asarray(frame, dtype=np.float64)
    if x.size == 0:
        return float("-inf")
    rms = np.sqrt(np.mean(np.square(x)))
    return float(20.0 * np.log10(rms + 1e-12))


def simple_vad(frame: NDArray[np.floating], sr: int, thr_db: float = -45.0) -> bool:
    """Simple energy-based VAD with optional zero-crossing check."""

    level = rms_db(frame)
    if not np.isfinite(level) or level < thr_db:
        return False

    x = np.asarray(frame, dtype=np.float64)
    if x.size < 2:
        return False

    signs = np.signbit(x)
    zcr = np.count_nonzero(signs[1:] != signs[:-1]) / (x.size - 1)
    max_zcr = 0.45
    if zcr > max_zcr:
        return False

    return True
