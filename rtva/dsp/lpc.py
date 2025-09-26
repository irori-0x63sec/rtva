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


def _pre_emphasis(x: NDArray[np.floating], coef: float = 0.97) -> NDArray[np.floating]:
    """Apply a simple pre-emphasis filter."""

    return lfilter([1.0, -coef], [1.0], x)


def _roots_to_formants(
    roots: NDArray[np.complexfloating], sr: int, fmax: int, n_formants: int = 5
) -> tuple[float, ...]:
    """Convert LPC polynomial roots to up to ``n_formants`` frequencies."""

    if roots.size == 0:
        return tuple(0.0 for _ in range(n_formants))

    usable = roots[(np.abs(roots) > 1e-6) & (np.abs(roots) < 1.0 + 1e-6)]
    usable = usable[np.imag(usable) >= 0]
    if usable.size == 0:
        return tuple(0.0 for _ in range(n_formants))

    ang = np.arctan2(np.imag(usable), np.real(usable))
    freqs = np.sort(ang * sr / (2.0 * np.pi))
    freqs = freqs[(freqs > 50.0) & (freqs < float(fmax))]

    out = np.zeros(max(1, int(n_formants)), dtype=float)
    limit = min(out.size, freqs.size)
    if limit:
        out[:limit] = freqs[:limit]
    return tuple(float(v) for v in out)


def lpc_formants(
    frame: NDArray[np.floating],
    sr: int,
    order: int = 16,
    fmax: int = 5500,
    n_formants: int = 5,
) -> tuple[float, ...]:
    """Estimate up to ``n_formants`` formants using LPC analysis."""

    if frame.size == 0:
        return tuple(0.0 for _ in range(n_formants))

    x = np.asarray(frame, dtype=np.float64)
    if not np.any(np.isfinite(x)):
        return tuple(0.0 for _ in range(n_formants))

    x = _pre_emphasis(x, 0.97)
    x *= np.hanning(x.size)

    if not np.any(np.abs(x) > 0):
        return tuple(0.0 for _ in range(n_formants))

    autocorr = np.correlate(x, x, mode="full")[x.size - 1 :]
    if autocorr.size <= order or autocorr[0] <= 0:
        return tuple(0.0 for _ in range(n_formants))

    r = autocorr[: order + 1]
    try:
        a_vec = solve_toeplitz((r[:-1], r[:-1]), r[1:])
    except LinAlgError:
        return tuple(0.0 for _ in range(n_formants))

    a = np.concatenate(([1.0], -a_vec))
    if not np.all(np.isfinite(a)):
        return tuple(0.0 for _ in range(n_formants))

    roots = np.roots(a)
    return _roots_to_formants(roots, sr, fmax, n_formants=n_formants)


def burg_formants_parselmouth(
    frame: NDArray[np.floating],
    sr: int,
    fmax: int = 5500,
    n_formants: int = 5,
) -> tuple[float, ...]:
    """Estimate formants using Praat's Burg method via parselmouth."""

    if parselmouth is None:
        return tuple(0.0 for _ in range(n_formants))

    try:
        snd = parselmouth.Sound(frame.astype(float), sampling_frequency=sr)
        formant = snd.to_formant_burg(
            time_step=0.0,
            max_number_of_formants=max(5, n_formants),
            maximum_formant=fmax,
            pre_emphasis_from=50.0,
        )
        t_mid = snd.xmin + 0.5 * snd.get_total_duration()
        values: list[float] = []
        for i in range(1, n_formants + 1):
            val = formant.get_value_at_time(i, t_mid)
            if val is None or math.isnan(val):
                values.append(0.0)
            else:
                values.append(float(val))
        while len(values) < n_formants:
            values.append(0.0)
        return tuple(values[:n_formants])
    except Exception:
        return tuple(0.0 for _ in range(n_formants))
