from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

SOURCE_COLORS = {
    "A": (255, 200, 0),
    "B": (0, 200, 255),
}

FORMANT_COLORS = [
    (255, 180, 0),
    (255, 110, 50),
    (210, 90, 255),
    (90, 200, 255),
    (120, 255, 140),
]


def _color_for(source: str) -> tuple[int, int, int]:
    return SOURCE_COLORS.get(source, (200, 200, 200))


class PitchPanel(QWidget):
    """Pitch contour panel with multi-source overlay."""

    def __init__(
        self, timespan_sec: int = 10, sr_hop: int = 100, sources: Sequence[str] = ("A", "B")
    ) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        head = QHBoxLayout()
        self.value_label = QLabel("F0: --")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.stability_label = QLabel("±cent: --")
        head.addWidget(self.value_label)
        head.addWidget(self.stability_label)
        layout.addLayout(head)

        self.plot = pg.PlotWidget()
        self.plot.setYRange(60, 600)
        self.plot.setLabel("left", "F0 (Hz)")
        self.plot.setLabel("bottom", "time (s)")
        self.plot.addLegend()
        layout.addWidget(self.plot)

        self.sources = tuple(sources)
        self.timespan_sec = float(timespan_sec)
        self.rate_hz = float(sr_hop)
        self.capacity = max(1, int(self.timespan_sec * self.rate_hz))

        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._buffers: Dict[str, np.ndarray] = {}
        self._indices: Dict[str, int] = {}
        self._last_values: Dict[str, float] = {src: float("nan") for src in self.sources}
        self._stability: Dict[str, float | None] = {src: None for src in self.sources}

        for src in self.sources:
            self._init_source(src)

    def _init_source(self, source: str) -> None:
        color = SOURCE_COLORS.get(source, (200, 200, 200))
        pen = pg.mkPen(color=color, width=2)
        curve = self.plot.plot(pen=pen, name=source)
        self._curves[source] = curve
        self._buffers[source] = np.full(self.capacity, np.nan, dtype=float)
        self._indices[source] = 0

    def _ensure_source(self, source: str) -> None:
        if source not in self._curves:
            self.sources += (source,)
            self._init_source(source)
            self._last_values[source] = float("nan")
            self._stability[source] = None

    def update_pitch(self, f0_hz: float, source: str = "A") -> None:
        self._ensure_source(source)
        value = f0_hz if f0_hz > 0 else np.nan
        buf = self._buffers[source]
        buf[self._indices[source] % self.capacity] = value
        self._indices[source] += 1
        y = buf.take(np.arange(self.capacity), mode="wrap")
        x = np.linspace(-len(y) / max(self.rate_hz, 1.0), 0.0, len(y))
        self._curves[source].setData(x, y)
        self._last_values[source] = float(value) if np.isfinite(value) else float("nan")
        self._refresh_labels()

    def set_series(self, source: str, y: np.ndarray | None, x: np.ndarray | None) -> None:
        self._ensure_source(source)
        if y is None or x is None or len(y) == 0 or len(x) == 0:
            self._curves[source].setData([], [])
            self._last_values[source] = float("nan")
        else:
            self._curves[source].setData(x, y)
            self._last_values[source] = float(y[-1]) if np.isfinite(y[-1]) else float("nan")
        self._refresh_labels()

    def update_stability(self, cents_std: float | None, source: str = "A") -> None:
        self._ensure_source(source)
        self._stability[source] = cents_std
        self._refresh_labels()

    def set_rate(self, rate_hz: float) -> None:
        if rate_hz <= 0:
            return
        self.rate_hz = float(rate_hz)
        capacity = max(1, int(self.timespan_sec * self.rate_hz))
        if capacity == self.capacity:
            return
        self.capacity = capacity
        for source in list(self._curves):
            self._buffers[source] = np.full(self.capacity, np.nan, dtype=float)
            self._indices[source] = 0
            self._curves[source].setData([], [])
        self._refresh_labels()

    def clear(self) -> None:
        for source in list(self._curves):
            self._buffers[source][:] = np.nan
            self._indices[source] = 0
            self._curves[source].setData([], [])
            self._last_values[source] = float("nan")
            self._stability[source] = None
        self._refresh_labels()

    def buffer_for(self, source: str) -> np.ndarray:
        self._ensure_source(source)
        return self._buffers[source].copy()

    def _refresh_labels(self) -> None:
        parts = []
        for source in self.sources:
            value = self._last_values.get(source, float("nan"))
            if np.isfinite(value) and value > 0:
                parts.append(f"{source}: {value:6.1f} Hz")
            else:
                parts.append(f"{source}: -- Hz")
        self.value_label.setText(" | ".join(parts) if parts else "F0: --")

        stab_parts = []
        for source in self.sources:
            st = self._stability.get(source)
            if st is None or (isinstance(st, float) and np.isnan(st)):
                stab_parts.append(f"{source}: --")
            else:
                stab_parts.append(f"{source}: {st:4.1f}")
        self.stability_label.setText(
            "±cent: " + " | ".join(stab_parts) if stab_parts else "±cent: --"
        )


class SpectroPanel(QWidget):
    """Spectrogram panel with layered sources and formant overlays."""

    def __init__(
        self,
        window_sec: float = 5.0,
        fmax_hz: float = 6000.0,
        sources: Sequence[str] = ("A", "B"),
        formant_count: int = 5,
    ) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot = pg.PlotWidget()
        self.plot.invertY(True)
        self.plot.setLabel("left", "Frequency (Hz)")
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setYRange(0, fmax_hz)
        self.plot.setLimits(yMin=0, yMax=fmax_hz)
        self.plot.addLegend()
        layout.addWidget(self.plot)

        self.window_sec = float(window_sec)
        self.fmax_hz = float(fmax_hz)
        self.sources = tuple(sources)
        self.formant_count = max(1, int(formant_count))

        self._images: Dict[str, pg.ImageItem] = {}
        self._last_layers: Dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]] | None
        ] = {}
        self._auto_levels: Dict[str, tuple[float, float]] = {
            src: (-80.0, 0.0) for src in self.sources
        }
        self._manual_levels: tuple[float, float] | None = None
        self._visible_formants: tuple[bool, ...] = tuple(True for _ in range(self.formant_count))
        self._formant_curves: Dict[str, list[pg.PlotDataItem]] = {}
        self._colormap_name = "magma"

        for source in self.sources:
            image = pg.ImageItem(axisOrder="row-major")
            if source == "B":
                image.setOpacity(0.6)
            self.plot.addItem(image)
            self._images[source] = image
            curves: list[pg.PlotDataItem] = []
            for idx in range(self.formant_count):
                color = FORMANT_COLORS[idx % len(FORMANT_COLORS)]
                pen = pg.mkPen(color=color, width=2)
                if source == "B":
                    pen = pg.mkPen(color=color, width=1, style=Qt.PenStyle.DashLine)
                curve = self.plot.plot(pen=pen, name=f"{source} F{idx + 1}")
                curves.append(curve)
            self._formant_curves[source] = curves
            self._last_layers[source] = None

        self.set_colormap(self._colormap_name)

    def set_colormap(self, name: str) -> None:
        self._colormap_name = name
        try:
            cmap = pg.colormap.get(name)
            lut = cmap.getLookupTable(alpha=True)
        except Exception:
            lut = None
        for image in self._images.values():
            if lut is not None:
                image.setLookupTable(lut)
            else:
                image.setLookupTable(None)
        self._repaint_layers()

    def set_db_levels(self, levels: tuple[float, float] | None) -> None:
        self._manual_levels = levels
        self._repaint_layers()

    def set_formant_visibility(self, visible: Iterable[bool]) -> None:
        flags = list(bool(v) for v in visible)
        if len(flags) < self.formant_count:
            flags.extend([True] * (self.formant_count - len(flags)))
        self._visible_formants = tuple(flags[: self.formant_count])
        for source in self.sources:
            self._render_formants(source)

    def clear(self) -> None:
        for source in self.sources:
            self._last_layers[source] = None
            image = self._images[source]
            image.setImage(np.zeros((1, 1), dtype=np.float32))
            image.setVisible(False)
            for curve in self._formant_curves[source]:
                curve.setData([], [])

    def set_layer(
        self,
        source: str,
        S_db: np.ndarray | None,
        freqs: np.ndarray | None,
        times: np.ndarray | None,
        formants: tuple[np.ndarray, ...] | None,
    ) -> None:
        if source not in self._images:
            return
        if (
            S_db is None
            or freqs is None
            or times is None
            or S_db.size == 0
            or freqs.size == 0
            or times.size == 0
        ):
            self._last_layers[source] = None
            self._images[source].setVisible(False)
            for curve in self._formant_curves[source]:
                curve.setData([], [])
            return

        S_db = np.asarray(S_db, dtype=np.float32)
        freqs = np.asarray(freqs, dtype=np.float32)
        times = np.asarray(times, dtype=np.float32)

        if times.size > 1:
            t_end = float(times[-1])
            t_start = t_end - self.window_sec
            mask_t = times >= t_start
        else:
            mask_t = np.ones_like(times, dtype=bool)

        if not np.any(mask_t):
            mask_t = np.ones_like(times, dtype=bool)

        times_masked = times[mask_t]
        S_db = S_db[:, mask_t]

        mask_f = freqs <= self.fmax_hz
        freqs_masked = freqs[mask_f]
        S_db = S_db[mask_f, :]

        if S_db.size == 0 or freqs_masked.size == 0 or times_masked.size == 0:
            self._last_layers[source] = None
            self._images[source].setVisible(False)
            return

        if formants is not None:
            processed_formants: list[np.ndarray] = []
            for idx in range(self.formant_count):
                if idx < len(formants):
                    arr = np.asarray(formants[idx], dtype=float)
                    if arr.size == times.size:
                        arr = arr[mask_t]
                    elif arr.size >= times_masked.size:
                        arr = arr[-times_masked.size :]
                    else:
                        arr = np.pad(
                            arr, (max(0, times_masked.size - arr.size), 0), constant_values=np.nan
                        )
                else:
                    arr = np.full(times_masked.size, np.nan, dtype=float)
                processed_formants.append(arr)
            formant_tuple = tuple(processed_formants)
        else:
            formant_tuple = tuple(
                np.full(times_masked.size, np.nan, dtype=float) for _ in range(self.formant_count)
            )

        self._last_layers[source] = (S_db, freqs_masked, times_masked, formant_tuple)
        self._render_layer(source)

    def _update_levels(
        self, source: str, S_db: np.ndarray, alpha: float = 0.2
    ) -> tuple[float, float]:
        vmin = float(np.nanpercentile(S_db, 5))
        vmax = float(np.nanpercentile(S_db, 98))
        if vmax - vmin < 10.0:
            vmin, vmax = vmin - 5.0, vmin + 5.0
        lv0, lv1 = self._auto_levels.get(source, (-80.0, 0.0))
        nv0 = (1 - alpha) * lv0 + alpha * vmin
        nv1 = (1 - alpha) * lv1 + alpha * vmax
        self._auto_levels[source] = (nv0, nv1)
        return self._auto_levels[source]

    def _render_layer(self, source: str) -> None:
        data = self._last_layers.get(source)
        image = self._images[source]
        if not data:
            image.setVisible(False)
            return
        S_db, freqs, times, formants = data
        levels = (
            self._manual_levels
            if self._manual_levels is not None
            else self._update_levels(source, S_db)
        )
        try:
            image.setImage(S_db, autoLevels=False, levels=levels)
        except Exception:
            return
        image.resetTransform()
        t0 = float(times[0])
        t1 = float(times[-1]) if times.size > 1 else t0 + 1e-3
        f0 = float(freqs[0])
        f1 = float(freqs[-1]) if freqs.size > 1 else f0 + 1.0
        image.setRect(QRectF(t0, f0, (t1 - t0), (f1 - f0)))
        image.setVisible(True)
        self._render_formants(source)

    def _render_formants(self, source: str) -> None:
        data = self._last_layers.get(source)
        if not data:
            for curve in self._formant_curves[source]:
                curve.setData([], [])
            return
        _, _, times, formants = data
        for idx, curve in enumerate(self._formant_curves[source]):
            if idx >= len(self._visible_formants) or not self._visible_formants[idx]:
                curve.setData([], [])
                continue
            arr = formants[idx]
            curve.setData(times, arr)

    def _repaint_layers(self) -> None:
        for source in self.sources:
            self._render_layer(source)


class HarmonicsPanel(QWidget):
    """Panel displaying H1–H2 and HNR trends for multiple sources."""

    def __init__(
        self, timespan_sec: float = 10.0, sr_hop: int = 50, sources: Sequence[str] = ("A", "B")
    ) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.value_label = QLabel("H1–H2: -- | HNR: --")
        layout.addWidget(self.value_label)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "dB")
        self.plot.setLabel("bottom", "time (s)")
        self.plot.setYRange(-60, 40)
        self.plot.addLegend()
        layout.addWidget(self.plot)

        self.sources = tuple(sources)
        self.timespan_sec = float(timespan_sec)
        self.rate_hz = float(sr_hop)
        self.capacity = max(1, int(self.timespan_sec * self.rate_hz))

        self._curves: Dict[str, dict[str, pg.PlotDataItem]] = {}
        self._buffers: Dict[str, dict[str, np.ndarray]] = {}
        self._indices: Dict[str, int] = {}
        self._last_values: Dict[str, tuple[float, float]] = {
            src: (float("nan"), float("nan")) for src in self.sources
        }

        for source in self.sources:
            self._init_source(source)

    def _init_source(self, source: str) -> None:
        color = _color_for(source)
        h1_pen = pg.mkPen(color=color, width=2)
        hnr_pen = pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine)
        curves = {
            "h1h2": self.plot.plot(pen=h1_pen, name=f"{source} H1-H2"),
            "hnr": self.plot.plot(pen=hnr_pen, name=f"{source} HNR"),
        }
        self._curves[source] = curves
        self._buffers[source] = {
            "h1h2": np.full(self.capacity, np.nan, dtype=float),
            "hnr": np.full(self.capacity, np.nan, dtype=float),
        }
        self._indices[source] = 0

    def _ensure_source(self, source: str) -> None:
        if source not in self._curves:
            self._init_source(source)
            self._last_values[source] = (float("nan"), float("nan"))

    def update_values(self, h1h2: float | None, hnr: float | None, source: str = "A") -> None:
        self._ensure_source(source)
        idx = self._indices[source] % self.capacity
        self._indices[source] += 1
        buf = self._buffers[source]
        buf["h1h2"][idx] = float(h1h2) if h1h2 is not None and np.isfinite(h1h2) else np.nan
        buf["hnr"][idx] = float(hnr) if hnr is not None and np.isfinite(hnr) else np.nan
        y = buf["h1h2"].take(np.arange(self.capacity), mode="wrap")
        x = np.linspace(-len(y) / max(self.rate_hz, 1.0), 0.0, len(y))
        self._curves[source]["h1h2"].setData(x, y)
        y2 = buf["hnr"].take(np.arange(self.capacity), mode="wrap")
        self._curves[source]["hnr"].setData(x, y2)
        self._last_values[source] = (
            float(buf["h1h2"][idx]) if np.isfinite(buf["h1h2"][idx]) else float("nan"),
            float(buf["hnr"][idx]) if np.isfinite(buf["hnr"][idx]) else float("nan"),
        )
        self._refresh_label()

    def set_series(
        self,
        source: str,
        times: np.ndarray | None,
        h1h2: np.ndarray | None,
        hnr: np.ndarray | None,
    ) -> None:
        self._ensure_source(source)
        if times is None or h1h2 is None or hnr is None or len(times) == 0 or len(h1h2) == 0:
            self._curves[source]["h1h2"].setData([], [])
            self._curves[source]["hnr"].setData([], [])
            self._last_values[source] = (float("nan"), float("nan"))
        else:
            self._curves[source]["h1h2"].setData(times, h1h2)
            self._curves[source]["hnr"].setData(times, hnr)
            self._last_values[source] = (
                float(h1h2[-1]) if np.isfinite(h1h2[-1]) else float("nan"),
                float(hnr[-1]) if np.isfinite(hnr[-1]) else float("nan"),
            )
        self._refresh_label()

    def set_rate(self, rate_hz: float) -> None:
        if rate_hz <= 0:
            return
        self.rate_hz = float(rate_hz)
        capacity = max(1, int(self.timespan_sec * self.rate_hz))
        if capacity == self.capacity:
            return
        self.capacity = capacity
        for source in list(self._curves):
            self._buffers[source] = {
                "h1h2": np.full(self.capacity, np.nan, dtype=float),
                "hnr": np.full(self.capacity, np.nan, dtype=float),
            }
            self._indices[source] = 0
        self._refresh_label()

    def clear(self) -> None:
        for source in list(self._curves):
            for key in self._buffers[source]:
                self._buffers[source][key][:] = np.nan
            self._indices[source] = 0
            self._curves[source]["h1h2"].setData([], [])
            self._curves[source]["hnr"].setData([], [])
            self._last_values[source] = (float("nan"), float("nan"))
        self._refresh_label()

    def _refresh_label(self) -> None:
        parts = []
        for source in self.sources:
            h1h2, hnr = self._last_values.get(source, (float("nan"), float("nan")))
            if np.isfinite(h1h2):
                h1 = f"{h1h2:5.2f}"
            else:
                h1 = "--"
            if np.isfinite(hnr):
                hn = f"{hnr:5.2f}"
            else:
                hn = "--"
            parts.append(f"{source} H1–H2: {h1} dB / HNR: {hn} dB")
        self.value_label.setText(" | ".join(parts) if parts else "H1–H2: -- | HNR: --")


class CPPPanel(QWidget):
    """Cepstral peak prominence panel with multi-source overlay."""

    def __init__(
        self, timespan_sec: float = 30.0, sr_hop: int = 10, sources: Sequence[str] = ("A", "B")
    ) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.value_label = QLabel("CPP: --")
        layout.addWidget(self.value_label)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "CPP (dB)")
        self.plot.setLabel("bottom", "time (s)")
        self.plot.setYRange(-10, 35)
        self.plot.addLegend()
        layout.addWidget(self.plot)

        self.sources = tuple(sources)
        self.timespan_sec = float(timespan_sec)
        self.rate_hz = float(sr_hop)
        self.capacity = max(1, int(self.timespan_sec * self.rate_hz))

        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._buffers: Dict[str, np.ndarray] = {}
        self._indices: Dict[str, int] = {}
        self._last_values: Dict[str, float] = {src: float("nan") for src in self.sources}

        for source in self.sources:
            self._init_source(source)

    def _init_source(self, source: str) -> None:
        pen = SOURCE_COLORS.get(source, (200, 200, 200))
        curve = self.plot.plot(pen=pg.mkPen(color=pen, width=2), name=source)
        self._curves[source] = curve
        self._buffers[source] = np.full(self.capacity, np.nan, dtype=float)
        self._indices[source] = 0

    def _ensure_source(self, source: str) -> None:
        if source not in self._curves:
            self._init_source(source)
            self._last_values[source] = float("nan")

    def update_value(self, value: float | None, source: str = "A") -> None:
        self._ensure_source(source)
        v = float(value) if value is not None and np.isfinite(value) else np.nan
        buf = self._buffers[source]
        buf[self._indices[source] % self.capacity] = v
        self._indices[source] += 1
        y = buf.take(np.arange(self.capacity), mode="wrap")
        x = np.linspace(-len(y) / max(self.rate_hz, 1.0), 0.0, len(y))
        self._curves[source].setData(x, y)
        self._last_values[source] = float(v) if np.isfinite(v) else float("nan")
        self._refresh_label()

    def set_series(self, source: str, times: np.ndarray | None, values: np.ndarray | None) -> None:
        self._ensure_source(source)
        if times is None or values is None or len(times) == 0:
            self._curves[source].setData([], [])
            self._last_values[source] = float("nan")
        else:
            self._curves[source].setData(times, values)
            self._last_values[source] = (
                float(values[-1]) if np.isfinite(values[-1]) else float("nan")
            )
        self._refresh_label()

    def set_rate(self, rate_hz: float) -> None:
        if rate_hz <= 0:
            return
        self.rate_hz = float(rate_hz)
        capacity = max(1, int(self.timespan_sec * self.rate_hz))
        if capacity == self.capacity:
            return
        self.capacity = capacity
        for source in list(self._curves):
            self._buffers[source] = np.full(self.capacity, np.nan, dtype=float)
            self._indices[source] = 0
        self._refresh_label()

    def clear(self) -> None:
        for source in list(self._curves):
            self._buffers[source][:] = np.nan
            self._indices[source] = 0
            self._curves[source].setData([], [])
            self._last_values[source] = float("nan")
        self._refresh_label()

    def _refresh_label(self) -> None:
        parts = []
        for source in self.sources:
            value = self._last_values.get(source, float("nan"))
            if np.isfinite(value):
                parts.append(f"{source}: {value:5.2f} dB")
            else:
                parts.append(f"{source}: -- dB")
        self.value_label.setText("CPP: " + " | ".join(parts) if parts else "CPP: -- dB")
