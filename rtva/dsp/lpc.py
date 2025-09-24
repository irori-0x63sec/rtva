"""Linear predictive coding helpers for formant analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter


def pre_emphasis(
    x: NDArray[np.float64] | NDArray[np.float32],
    coef: float = 0.97,
) -> NDArray[np.float64]:
    """Apply a simple pre-emphasis filter to the signal."""

    return lfilter([1.0, -coef], [1.0], x).astype(np.float64, copy=False)


def _levinson_durbin(r: NDArray[np.float64], order: int) -> tuple[NDArray[np.float64], float]:
    """Solve the Yule-Walker equations via Levinson-Durbin recursion."""

    a = np.zeros(order + 1, dtype=np.float64)
    e = r[0]
    if e <= 0.0:
        return a, float(e)

    a[0] = 1.0
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / e
        a_prev = a.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = a_prev[j] + k * a_prev[i - j]
        e *= 1.0 - k * k
        if e <= 0.0:
            e = 1e-9
            break
    return a, float(e)


def lpc_formants(
    frame: NDArray[np.float32] | NDArray[np.float64],
    sr: int,
    order: int = 14,
    fmax: int = 5000,
) -> tuple[float, float, float]:
    """Estimate the first three formants in Hz using LPC analysis.

    Args:
        frame: Single analysis frame (windowed) of audio samples.
        sr: Sampling rate in Hz.
        order: LPC order (usually 2 * expected number of formants + 2).
        fmax: Upper frequency limit for reported formants.

    Returns:
        Tuple of (F1, F2, F3) in Hz. Missing formants are reported as 0.0.
    """

    if frame.size == 0 or sr <= 0:
        return 0.0, 0.0, 0.0

    order = int(order)
    if order < 2:
        return 0.0, 0.0, 0.0

    x = np.asarray(frame, dtype=np.float64)
    if not np.any(x):
        return 0.0, 0.0, 0.0

    x = pre_emphasis(x, 0.97)
    x -= np.mean(x)
    if not np.any(np.abs(x) > 0):
        return 0.0, 0.0, 0.0

    n = len(x)
    if order >= n:
        order = n - 1
    if order < 2:
        return 0.0, 0.0, 0.0

    autocorr = np.correlate(x, x, mode="full")
    autocorr = autocorr[n - 1 : n + order]
    if autocorr[0] <= 0.0:
        return 0.0, 0.0, 0.0

    a, _ = _levinson_durbin(autocorr, order)
    if not np.any(a):
        return 0.0, 0.0, 0.0

    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0.0]
    ang = np.arctan2(np.imag(roots), np.real(roots))
    freqs = ang * (sr / (2.0 * np.pi))
    freqs = freqs[(freqs > 0.0) & (freqs < fmax)]
    if freqs.size == 0:
        return 0.0, 0.0, 0.0

    freqs.sort()
    f1 = float(freqs[0]) if freqs.size > 0 else 0.0
    f2 = float(freqs[1]) if freqs.size > 1 else 0.0
    f3 = float(freqs[2]) if freqs.size > 2 else 0.0
    return f1, f2, f3
