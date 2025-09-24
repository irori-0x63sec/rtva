from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth import PraatError
from scipy.signal import lfilter


def pre_emphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return lfilter([1, -coef], [1], x)


def _levinson_durbin(autocorr: np.ndarray, order: int) -> np.ndarray | None:
    """Return LPC coefficients using Levinson-Durbin recursion."""

    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    error = float(autocorr[0])
    if error <= 0:
        return None

    for i in range(1, order + 1):
        acc = autocorr[i]
        for j in range(1, i):
            acc -= a[j] * autocorr[i - j]

        if abs(error) < 1e-12:
            return None

        reflection = acc / error
        prev = a.copy()
        for j in range(1, i):
            a[j] = prev[j] - reflection * prev[i - j]
        a[i] = reflection
        error *= 1.0 - reflection * reflection
        if error <= 1e-12:
            return None

    return a


def _roots_to_formants(roots: np.ndarray, sr: int, fmax: int) -> tuple[float, float, float]:
    candidates: list[tuple[float, float]] = []
    for r in roots:
        if np.imag(r) <= 0:
            continue
        magnitude = np.abs(r)
        if magnitude == 0:
            continue
        angle = np.arctan2(np.imag(r), np.real(r))
        freq = angle * (sr / (2 * np.pi))
        bandwidth = -0.5 * sr * np.log(magnitude)
        if 0 < freq < fmax and bandwidth < 500:
            candidates.append((float(freq), float(bandwidth)))

    candidates.sort(key=lambda item: item[0])
    freqs = [freq for freq, _ in candidates[:3]]
    while len(freqs) < 3:
        freqs.append(0.0)
    return freqs[0], freqs[1], freqs[2]


def lpc_formants(
    frame: np.ndarray, sr: int, order: int = 14, fmax: int = 5000
) -> tuple[float, float, float]:
    """
    Pre-emphasis → Hann window → LPC → polynomial roots → formant frequencies.
    Returns (F1, F2, F3) in Hz. Returns zeros on failure.
    """

    if frame.size == 0:
        return 0.0, 0.0, 0.0

    signal = np.asarray(frame, dtype=np.float64)
    signal = signal - np.mean(signal)
    emphasized = pre_emphasis(signal)
    windowed = emphasized * np.hanning(len(emphasized))

    if not np.any(windowed):
        return 0.0, 0.0, 0.0

    autocorr = np.correlate(windowed, windowed, mode="full")
    mid = autocorr.size // 2
    r = autocorr[mid : mid + order + 1]

    coeffs = _levinson_durbin(r, order)
    if coeffs is None:
        return 0.0, 0.0, 0.0

    try:
        roots = np.roots(np.append(1.0, -coeffs[1:]))
    except (ValueError, np.linalg.LinAlgError):
        return 0.0, 0.0, 0.0

    return _roots_to_formants(roots, sr, fmax)


def burg_formants_parselmouth(
    frame: np.ndarray, sr: int, fmax: int = 5500
) -> tuple[float, float, float]:
    """Praat Burg formant estimation via parselmouth."""

    if frame.size == 0:
        return 0.0, 0.0, 0.0

    samples = frame.astype(np.float64)
    samples -= np.mean(samples)
    sound = parselmouth.Sound(samples, sampling_frequency=sr)
    window_length = min(0.04, len(frame) / sr)
    try:
        formant = sound.to_formant_burg(
            time_step=None,
            max_number_of_formants=5,
            maximum_formant=fmax,
            window_length=window_length,
            pre_emphasis_from=50.0,
        )
    except PraatError:
        return 0.0, 0.0, 0.0

    time = sound.xmin + sound.duration / 2
    candidates: list[tuple[float, float]] = []
    for i in range(1, 7):
        freq = formant.get_value_at_time(i, time)
        bandwidth = formant.get_bandwidth_at_time(i, time)
        if (
            freq is None
            or bandwidth is None
            or np.isnan(freq)
            or np.isnan(bandwidth)
            or freq <= 0
            or freq >= fmax
        ):
            continue
        candidates.append((float(freq), float(bandwidth)))

    if not candidates:
        return 0.0, 0.0, 0.0

    selected: list[float] = []
    ranges = [
        (80.0, min(1200.0, float(fmax))),
        (800.0, min(3000.0, float(fmax))),
        (1800.0, float(fmax)),
    ]
    remaining = sorted(candidates, key=lambda item: item[0])
    for low, high in ranges:
        chosen = None
        for idx, item in enumerate(remaining):
            if low <= item[0] <= high:
                chosen = remaining.pop(idx)[0]
                break
        if chosen is None:
            selected.append(0.0)
        else:
            selected.append(chosen)

    while len(selected) < 3:
        selected.append(0.0)

    return tuple(selected[:3])
