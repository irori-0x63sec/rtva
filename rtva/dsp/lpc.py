from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, solve_toeplitz
from scipy.signal import lfilter

try:  # Praat bindings are optional in some environments
    import parselmouth
except Exception:  # pragma: no cover - parselmouth may be missing
    parselmouth = None  # type: ignore


def pre_emphasis(x: NDArray[np.floating], coef: float = 0.97) -> NDArray[np.floating]:
    """Apply a simple pre-emphasis filter."""

    return lfilter([1.0, -coef], [1.0], x)


def _roots_to_formants(
    roots: NDArray[np.complexfloating], sr: int, fmax: int
) -> tuple[float, float, float]:
    """Convert LPC polynomial roots to up to three formant frequencies."""

    if roots.size == 0:
        return (0.0, 0.0, 0.0)

    usable = roots[(np.abs(roots) > 1e-6) & (np.abs(roots) < 1.0 + 1e-6)]
    usable = usable[np.imag(usable) >= 0]
    if usable.size == 0:
        return (0.0, 0.0, 0.0)

    ang = np.arctan2(np.imag(usable), np.real(usable))
    freqs = np.sort(ang * sr / (2.0 * np.pi))
    freqs = freqs[(freqs > 50.0) & (freqs < float(fmax))]

    out = np.zeros(3, dtype=float)
    limit = min(3, freqs.size)
    if limit:
        out[:limit] = freqs[:limit]
    return float(out[0]), float(out[1]), float(out[2])


def lpc_formants(
    frame: NDArray[np.floating], sr: int, order: int = 14, fmax: int = 5000
) -> tuple[float, float, float]:
    """Estimate up to the first three formants using LPC analysis."""

    if frame.size == 0:
        return (0.0, 0.0, 0.0)

    x = np.asarray(frame, dtype=np.float64)
    x = pre_emphasis(x, 0.97)
    x *= np.hanning(x.size)

    if not np.any(np.abs(x) > 0):
        return (0.0, 0.0, 0.0)

    autocorr = np.correlate(x, x, mode="full")[x.size - 1 :]
    if autocorr.size <= order or autocorr[0] <= 0:
        return (0.0, 0.0, 0.0)

    r = autocorr[: order + 1]
    try:
        a_vec = solve_toeplitz((r[:-1], r[:-1]), r[1:])
    except LinAlgError:
        return (0.0, 0.0, 0.0)

    a = np.concatenate(([1.0], -a_vec))
    if not np.all(np.isfinite(a)):
        return (0.0, 0.0, 0.0)

    roots = np.roots(a)
    return _roots_to_formants(roots, sr, fmax)


def burg_formants_parselmouth(
    frame: NDArray[np.floating], sr: int, fmax: int = 5500
) -> tuple[float, float, float]:
    """Estimate formants using Praat's Burg method via parselmouth."""

    if parselmouth is None:
        raise RuntimeError("parselmouth is not available")

    snd = parselmouth.Sound(frame.astype(float), sampling_frequency=sr)
    formant = snd.to_formant_burg(
        time_step=0.0,
        max_number_of_formants=5,
        maximum_formant=fmax,
        pre_emphasis_from=50.0,
    )
    t_mid = snd.xmin + 0.5 * snd.get_total_duration()
    values: list[float] = []
    for i in range(1, 4):
        val = formant.get_value_at_time(i, t_mid)
        if val is None or math.isnan(val):
            values.append(0.0)
        else:
            values.append(float(val))
    return values[0], values[1], values[2]
