from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Sequence

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from rtva.audio import AudioStream, hann
from rtva.config import AnalyzerConfig, save_preset
from rtva.dsp.cepstrum import cpp_db
from rtva.dsp.lpc import burg_formants_parselmouth, lpc_formants
from rtva.dsp.pitch import yin_f0
from rtva.dsp.spectrum import h1_h2_db, hnr_db, stft_mag_db
from rtva.dsp.vad import rms_db, simple_vad
from rtva.logging.recorder import ParquetRecorder
from rtva.ui.controls import ControlsPanel
from rtva.ui.panels import CPPPanel, HarmonicsPanel, PitchPanel, SpectroPanel

try:  # Praat bindings availability
    import parselmouth  # noqa: F401

    PARSELMOUTH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PARSELMOUTH_AVAILABLE = False

try:  # optional MP3 support
    from pydub import AudioSegment
except Exception:  # pragma: no cover - optional dependency
    AudioSegment = None  # type: ignore[assignment]


SESSIONS_DIR = Path("sessions")
SOURCES: Sequence[str] = ("A", "B")


@dataclass
class SourceState:
    name: str
    buffer: np.ndarray
    formant_history: list[Deque[float]]
    recording: list[np.ndarray] = field(default_factory=list)
    last_cpp: float = float("nan")
    last_h1h2: float = float("nan")
    last_hnr: float = float("nan")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RTVA — Real-Time Voice Analyzer")

        self.cfg = AnalyzerConfig()
        self._cpp_rate_hz = 2.0
        self._cpp_stride = 1
        self._cpp_counter = 0
        self._warn_burg_once = False

        self._parselmouth_available = PARSELMOUTH_AVAILABLE

        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.toolbar = QToolBar("Session", self)
        self.addToolBar(self.toolbar)

        self.action_open = QAction("Open…", self)
        self.action_clear = QAction("Clear", self)
        self.action_export_wav = QAction("Save WAV", self)
        self.action_export_png = QAction("Export PNG", self)
        self.action_play = QAction("Play", self)
        self.action_stop_play = QAction("Stop Playback", self)
        self.action_stop_play.setEnabled(False)

        self.toolbar.addAction(self.action_open)
        self.toolbar.addAction(self.action_clear)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_play)
        self.toolbar.addAction(self.action_stop_play)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_export_wav)
        self.toolbar.addAction(self.action_export_png)
        self.toolbar.addSeparator()

        self.source_combo = QComboBox()
        self.source_combo.addItem("A — Recording", "A")
        self.source_combo.addItem("B — File", "B")
        self.source_combo.setCurrentIndex(1)
        combo_action = QWidgetAction(self)
        combo_action.setDefaultWidget(self.source_combo)
        self.toolbar.addAction(combo_action)

        controls_bar = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        controls_bar.addWidget(self.btn_start)
        controls_bar.addWidget(self.btn_stop)
        controls_bar.addStretch(1)
        root_layout.addLayout(controls_bar)

        main_layout = QHBoxLayout()
        self.controls = ControlsPanel(self.cfg, parselmouth_available=self._parselmouth_available)
        main_layout.addWidget(self.controls, stretch=0)

        panels = QWidget(self)
        panels_layout = QVBoxLayout()
        panels.setLayout(panels_layout)
        self._panels_widget = panels

        self.pitch = PitchPanel(
            timespan_sec=10, sr_hop=int(round(1000 / self.cfg.hop_ms)), sources=SOURCES
        )
        self.spectro = SpectroPanel(
            window_sec=5.0, sources=SOURCES, formant_count=self.cfg.formant_count
        )
        self.harm = HarmonicsPanel(
            timespan_sec=10, sr_hop=int(round(1000 / self.cfg.hop_ms)), sources=SOURCES
        )
        self.cpp = CPPPanel(
            timespan_sec=30, sr_hop=int(max(1, round(self._cpp_rate_hz))), sources=SOURCES
        )

        panels_layout.addWidget(self.pitch)
        panels_layout.addWidget(self.spectro)
        panels_layout.addWidget(self.harm)
        panels_layout.addWidget(self.cpp)
        main_layout.addWidget(panels, stretch=1)
        root_layout.addLayout(main_layout, stretch=1)

        self.controls.config_changed.connect(self._apply_config)

        self.stream: Optional[AudioStream] = None
        self.running = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.recorder = ParquetRecorder(SESSIONS_DIR, rate_hz=self.cfg.log_rate_hz)
        self.recorder.start()

        self._spectrogram_window_sec = 5.0
        self._n_fft = 4096

        self._analysis_window = np.ones(1, dtype=np.float32)
        self._formant_history_len = 1
        self._sources: Dict[str, SourceState] = {}

        self._play_stream: Optional[sd.OutputStream] = None
        self._playback_data: Optional[np.ndarray] = None
        self._playback_pos = 0
        self._take_counter: Dict[str, int] = {name: 1 for name in SOURCES}

        self._apply_config(self.cfg)
        self._save_session_settings()

        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.action_open.triggered.connect(self._open_file_dialog)
        self.action_clear.triggered.connect(self.clear_all)
        self.action_export_wav.triggered.connect(self._export_wav_action)
        self.action_export_png.triggered.connect(self.export_snapshot_png)
        self.action_play.triggered.connect(self._play_action)
        self.action_stop_play.triggered.connect(self.stop_playback)

    def closeEvent(self, event):  # type: ignore[override]
        self.stop_analysis()
        self.stop_playback()
        self.recorder.stop()
        return super().closeEvent(event)

    def start_analysis(self) -> None:
        if self.running:
            return

        hop_samples = max(1, int(round(self.cfg.sr * self.cfg.hop_ms / 1000)))
        try:
            stream = AudioStream(sr=self.cfg.sr, blocksize=hop_samples)
            stream.start()
        except Exception as exc:  # pragma: no cover - runtime device errors
            if not hasattr(self, "_stream_err_once"):
                print(f"[WARN] failed to start audio stream: {exc}")
                self._stream_err_once = True
            self.stream = None
            return

        self.stream = stream
        self._reset_source("A", preserve_recording=False)
        self.running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start(self.cfg.hop_ms)

    def stop_analysis(self) -> None:
        if not self.running and self.stream is None:
            return

        try:
            if self.timer.isActive():
                self.timer.stop()
        except Exception:
            pass

        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
        self.stream = None
        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def clear_all(self) -> None:
        for source in SOURCES:
            self._reset_source(source, preserve_recording=False)
        self.pitch.clear()
        self.harm.clear()
        self.cpp.clear()
        self.spectro.clear()
        self._cpp_counter = 0

    def load_audio_file(self, path: str, target: str | None = None) -> None:
        if not path:
            return
        source = target or str(self.source_combo.currentData() or "B")
        if source not in SOURCES:
            source = "B"
        try:
            data, sr = self._read_audio(path)
        except Exception as exc:
            QMessageBox.warning(self, "Audio load failed", f"Failed to load audio: {exc}")
            return

        if data.size == 0:
            QMessageBox.information(self, "Audio load", "Audio file was empty")
            return

        if sr != self.cfg.sr:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=self.cfg.sr)
            sr = self.cfg.sr
        data = np.asarray(data, dtype=np.float32)

        self._sources[source].recording = [data.copy()]
        self._set_ring_buffer(source, data)
        self._analyze_static(source, data, sr)

    def export_wav(self, target: str | None = None) -> Optional[Path]:
        source = target or str(self.source_combo.currentData() or "A")
        if source not in SOURCES:
            source = "A"
        data = self._get_recording(source)
        if data.size == 0:
            return None

        session_dir = self._session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        take_idx = self._take_counter[source]
        self._take_counter[source] += 1
        filename = f"take_{take_idx:02d}_{source}.wav"
        path = session_dir / filename
        sf.write(path, data, self.cfg.sr)
        return path

    def export_snapshot_png(self) -> Optional[Path]:
        session_dir = self._session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            path = session_dir / f"snapshot_{index:02d}.png"
            if not path.exists():
                break
            index += 1
        pixmap = self._panels_widget.grab()
        if not pixmap.save(str(path)):
            return None
        return path

    def play_buffer(self, target: str | None = None) -> None:
        source = target or str(self.source_combo.currentData() or "A")
        if source not in SOURCES:
            source = "A"
        data = self._get_recording(source)
        if data.size == 0:
            return

        self.stop_playback()
        self._playback_data = data.astype(np.float32, copy=False)
        self._playback_pos = 0
        try:
            stream = sd.OutputStream(
                samplerate=self.cfg.sr,
                channels=1,
                dtype="float32",
                callback=self._playback_callback,
                finished_callback=self._on_playback_finished,
            )
            stream.start()
        except Exception as exc:  # pragma: no cover - device errors
            if not hasattr(self, "_playback_err_once"):
                print(f"[WARN] playback failed: {exc}")
                self._playback_err_once = True
            self._play_stream = None
            return

        self._play_stream = stream
        self.action_play.setEnabled(False)
        self.action_stop_play.setEnabled(True)

    def stop_playback(self) -> None:
        if self._play_stream is not None:
            try:
                self._play_stream.stop()
            except Exception:
                pass
            try:
                self._play_stream.close()
            except Exception:
                pass
        self._play_stream = None
        self._playback_data = None
        self._playback_pos = 0
        self.action_play.setEnabled(True)
        self.action_stop_play.setEnabled(False)

    def _playback_callback(self, outdata, frames, time_info, status):  # pragma: no cover - realtime
        if status:
            pass
        if self._playback_data is None:
            outdata.fill(0)
            raise sd.CallbackStop
        end = self._playback_pos + frames
        chunk = self._playback_data[self._playback_pos : end]
        outdata[: chunk.size, 0] = chunk
        if chunk.size < frames:
            outdata[chunk.size :, 0] = 0.0
            self._playback_pos = self._playback_data.size
            raise sd.CallbackStop
        self._playback_pos = end

    def _on_playback_finished(self):  # pragma: no cover - realtime
        self.stop_playback()

    def _apply_config(self, cfg: AnalyzerConfig) -> None:
        if cfg.formant_method == "burg" and not self._parselmouth_available:
            cfg = replace(cfg, formant_method="lpc")
            self.controls.set_config(cfg)

        was_running = self.running
        if was_running:
            self.stop_analysis()

        self.cfg = cfg
        self._cpp_stride = max(1, int(round((1000.0 / cfg.hop_ms) / self._cpp_rate_hz)))
        self._cpp_counter = 0

        hop_samples = max(1, int(round(cfg.sr * cfg.hop_ms / 1000)))
        frame_samples = max(hop_samples, int(round(cfg.sr * cfg.frame_ms / 1000)))
        self._n_fft = max(2048, 1 << int(np.ceil(np.log2(frame_samples))))

        buffer_frames = max(1, int(self._spectrogram_window_sec * 1000 / cfg.hop_ms))
        buffer_len = frame_samples + max(0, buffer_frames - 1) * hop_samples
        self._analysis_window = hann(frame_samples).astype(np.float32)
        self._formant_history_len = buffer_frames

        self._sources = {
            name: SourceState(
                name=name,
                buffer=np.zeros(buffer_len, dtype=np.float32),
                formant_history=self._init_formant_history(buffer_frames),
            )
            for name in SOURCES
        }

        rate_hz = 1000.0 / cfg.hop_ms
        self.pitch.set_rate(rate_hz)
        self.harm.set_rate(rate_hz)
        self.cpp.set_rate(self._cpp_rate_hz)

        self.spectro.set_colormap(cfg.colormap)
        if cfg.db_min is None or cfg.db_max is None:
            self.spectro.set_db_levels(None)
        else:
            self.spectro.set_db_levels((cfg.db_min, cfg.db_max))
        self.spectro.set_formant_visibility(cfg.show_formants)

        self.timer.setInterval(cfg.hop_ms)
        self.recorder.rate_hz = cfg.log_rate_hz
        self._save_session_settings()

        if was_running:
            self.start_analysis()

    def _init_formant_history(self, length: int) -> list[Deque[float]]:
        length = max(1, int(length))
        return [deque([np.nan] * length, maxlen=length) for _ in range(self.cfg.formant_count)]

    def _reset_source(self, source: str, preserve_recording: bool = False) -> None:
        state = self._sources[source]
        if state.buffer.size:
            state.buffer[:] = 0.0
        state.formant_history = self._init_formant_history(self._formant_history_len)
        state.last_cpp = float("nan")
        state.last_h1h2 = float("nan")
        state.last_hnr = float("nan")
        if not preserve_recording:
            state.recording = []

    def _append_audio(self, source: str, hop: np.ndarray) -> None:
        state = self._sources[source]
        hop = hop.astype(np.float32, copy=False)
        n = hop.size
        if n >= state.buffer.size:
            state.buffer[:] = hop[-state.buffer.size :]
        else:
            state.buffer = np.roll(state.buffer, -n)
            state.buffer[-n:] = hop
        state.recording.append(hop.copy())

    def _current_frame(self, source: str) -> np.ndarray:
        state = self._sources[source]
        frame_len = self._analysis_window.size
        return state.buffer[-frame_len:]

    def _tick(self) -> None:
        if not self.running or self.stream is None:
            return

        try:
            hop = self.stream.read()
        except Exception:
            return

        self._append_audio("A", hop)
        frame = (self._current_frame("A") * self._analysis_window).astype(np.float32)

        rms = rms_db(frame)
        voiced = simple_vad(frame, self.cfg.sr)

        f0 = yin_f0(frame, self.cfg.sr, fmin=self.cfg.pitch_min, fmax=self.cfg.pitch_max)
        if not voiced:
            f0 = 0.0
        self.pitch.update_pitch(f0, source="A")
        self.pitch.update_stability(self._cents_std_recent(self.pitch.buffer_for("A")), source="A")

        formant_vals = self._estimate_formants(frame)
        history = self._sources["A"].formant_history
        for idx in range(min(len(history), len(formant_vals))):
            value = formant_vals[idx]
            history[idx].append(float(value) if value and value > 0 else np.nan)

        h1h2_val = np.nan
        hnr_val = np.nan
        if voiced and f0 > 0:
            h1h2_val = h1_h2_db(frame, self.cfg.sr, f0)
            hnr_val = hnr_db(frame, self.cfg.sr, f0)
        self.harm.update_values(h1h2_val, hnr_val, source="A")
        self._sources["A"].last_h1h2 = float(h1h2_val) if np.isfinite(h1h2_val) else float("nan")
        self._sources["A"].last_hnr = float(hnr_val) if np.isfinite(hnr_val) else float("nan")

        if voiced and (self._cpp_counter % self._cpp_stride == 0):
            cpp_val = cpp_db(frame, self.cfg.sr)
            self._sources["A"].last_cpp = cpp_val
            self.cpp.update_value(cpp_val, source="A")
        elif self._cpp_counter % self._cpp_stride == 0:
            self._sources["A"].last_cpp = float("nan")
            self.cpp.update_value(float("nan"), source="A")
        self._cpp_counter += 1

        self._update_spectrogram("A")

        self._push_log(
            source="A",
            f0=f0,
            formants=formant_vals,
            h1h2_val=h1h2_val,
            hnr_val=hnr_val,
            cpp_val=self._sources["A"].last_cpp,
            rms=rms,
            voiced=voiced,
        )

    def _estimate_formants(self, frame: np.ndarray) -> Sequence[float]:
        method = self.cfg.formant_method
        n_formants = max(1, self.cfg.formant_count)
        values: Sequence[float]
        if method == "burg" and self._parselmouth_available:
            values = burg_formants_parselmouth(
                frame,
                self.cfg.sr,
                fmax=self.cfg.formant_max_hz,
                n_formants=n_formants,
            )
            if all(v == 0.0 for v in values):
                values = lpc_formants(
                    frame,
                    self.cfg.sr,
                    order=self.cfg.lpc_order,
                    fmax=self.cfg.formant_max_hz,
                    n_formants=n_formants,
                )
        elif method == "lpc":
            values = lpc_formants(
                frame,
                self.cfg.sr,
                order=self.cfg.lpc_order,
                fmax=self.cfg.formant_max_hz,
                n_formants=n_formants,
            )
        else:  # auto
            values = burg_formants_parselmouth(
                frame,
                self.cfg.sr,
                fmax=self.cfg.formant_max_hz,
                n_formants=n_formants,
            )
            if not self._parselmouth_available or all(v == 0.0 for v in values):
                values = lpc_formants(
                    frame,
                    self.cfg.sr,
                    order=self.cfg.lpc_order,
                    fmax=self.cfg.formant_max_hz,
                    n_formants=n_formants,
                )
                if not self._warn_burg_once and self.cfg.formant_method != "lpc":
                    print("[WARN] parselmouth unavailable, falling back to LPC formants")
                    self._warn_burg_once = True
        return values

    def _update_spectrogram(self, source: str) -> None:
        state = self._sources[source]
        try:
            S_db, freqs, times = stft_mag_db(
                state.buffer,
                self.cfg.sr,
                n_fft=self._n_fft,
                hop=max(1, int(round(self.cfg.sr * self.cfg.hop_ms / 1000))),
                win=self._analysis_window,
            )
        except Exception:
            return

        formants = self._formant_arrays(state.formant_history, times.size)
        self.spectro.set_layer(source, S_db, freqs, times, formants)

    def _formant_arrays(
        self, history: Iterable[Deque[float]], length: int
    ) -> tuple[np.ndarray, ...]:
        arrays = []
        for series in history:
            hist = np.array(series, dtype=float)
            if hist.size < length:
                padded = np.full(length, np.nan, dtype=float)
                padded[-hist.size :] = hist
            else:
                padded = hist[-length:]
            arrays.append(padded)
        while len(arrays) < self.cfg.formant_count:
            arrays.append(np.full(length, np.nan, dtype=float))
        return tuple(arrays)

    def _cents_std_recent(self, buffer: np.ndarray, n: int = 50) -> Optional[float]:
        vals = buffer[~np.isnan(buffer)]
        if len(vals) < max(5, n // 2):
            return None
        recent = vals[-n:]
        hz = recent[recent > 0]
        if len(hz) < 5:
            return None
        med = np.median(hz)
        if med <= 0:
            return None
        cents = 1200 * np.log2(hz / med)
        return float(np.std(cents))

    def _cents_std_from_series(self, series: np.ndarray) -> Optional[float]:
        vals = series[np.isfinite(series) & (series > 0)]
        if vals.size < 5:
            return None
        med = np.median(vals)
        if med <= 0:
            return None
        cents = 1200 * np.log2(vals / med)
        return float(np.std(cents))

    def _analyze_static(self, source: str, signal: np.ndarray, sr: int) -> None:
        hop_samples = max(1, int(round(sr * self.cfg.hop_ms / 1000)))
        frame_samples = self._analysis_window.size
        if signal.size < frame_samples:
            signal = np.pad(signal, (frame_samples - signal.size, 0))

        positions = list(range(0, max(signal.size - frame_samples + 1, 1), hop_samples))
        if not positions:
            positions = [0]

        n_frames = len(positions)
        pitch_vals = np.full(n_frames, np.nan, dtype=float)
        formants = [np.full(n_frames, np.nan, dtype=float) for _ in range(self.cfg.formant_count)]
        h1h2_vals = np.full(n_frames, np.nan, dtype=float)
        hnr_vals = np.full(n_frames, np.nan, dtype=float)
        cpp_vals = np.full(n_frames, np.nan, dtype=float)

        for idx, start in enumerate(positions):
            frame = signal[start : start + frame_samples]
            if frame.size < frame_samples:
                frame = np.pad(frame, (0, frame_samples - frame.size))
            frame = (frame * self._analysis_window).astype(np.float32)

            voiced = simple_vad(frame, sr)
            f0 = yin_f0(frame, sr, fmin=self.cfg.pitch_min, fmax=self.cfg.pitch_max)
            if not voiced:
                f0 = 0.0
            pitch_vals[idx] = f0 if f0 > 0 else np.nan

            if voiced:
                vals = self._estimate_formants(frame)
                for jdx in range(min(len(vals), len(formants))):
                    formants[jdx][idx] = vals[jdx] if vals[jdx] > 0 else np.nan
                if f0 > 0:
                    h1h2_vals[idx] = h1_h2_db(frame, sr, f0)
                    hnr_vals[idx] = hnr_db(frame, sr, f0)
                cpp_vals[idx] = cpp_db(frame, sr)

        times = np.linspace(-(n_frames - 1) * hop_samples / sr, 0.0, n_frames)
        S_db, freqs, spec_times = stft_mag_db(
            signal,
            sr,
            n_fft=self._n_fft,
            hop=hop_samples,
            win=self._analysis_window,
        )

        self.pitch.set_series(source, pitch_vals, times)
        self.pitch.update_stability(self._cents_std_from_series(pitch_vals), source=source)
        self.harm.set_series(source, times, h1h2_vals, hnr_vals)
        self.cpp.set_series(source, times, cpp_vals)
        self.spectro.set_layer(source, S_db, freqs, spec_times, tuple(formants))

        state = self._sources[source]
        state.formant_history = [
            deque(series[-self._formant_history_len :], maxlen=self._formant_history_len)
            for series in formants
        ]

    def _get_recording(self, source: str) -> np.ndarray:
        chunks = self._sources[source].recording
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _set_ring_buffer(self, source: str, data: np.ndarray) -> None:
        state = self._sources[source]
        if data.size >= state.buffer.size:
            state.buffer[:] = data[-state.buffer.size :]
        else:
            state.buffer[:] = 0.0
            state.buffer[-data.size :] = data

    def _push_log(
        self,
        source: str,
        f0: float,
        formants: Sequence[float],
        h1h2_val: float,
        hnr_val: float,
        cpp_val: float,
        rms: float,
        voiced: bool,
    ) -> None:
        if self.recorder is None:
            return
        formant_list = list(formants)
        while len(formant_list) < 5:
            formant_list.append(0.0)
        if len(formant_list) > 5:
            formant_list = formant_list[:5]
        row = {
            "timestamp_ns": time.time_ns(),
            "source": source,
            "f0_hz": float(f0) if f0 > 0 else np.nan,
            "f1_hz": (
                float(formant_list[0])
                if np.isfinite(formant_list[0]) and formant_list[0] > 0
                else np.nan
            ),
            "f2_hz": (
                float(formant_list[1])
                if np.isfinite(formant_list[1]) and formant_list[1] > 0
                else np.nan
            ),
            "f3_hz": (
                float(formant_list[2])
                if np.isfinite(formant_list[2]) and formant_list[2] > 0
                else np.nan
            ),
            "f4_hz": (
                float(formant_list[3])
                if np.isfinite(formant_list[3]) and formant_list[3] > 0
                else np.nan
            ),
            "f5_hz": (
                float(formant_list[4])
                if np.isfinite(formant_list[4]) and formant_list[4] > 0
                else np.nan
            ),
            "h1_h2_db": float(h1h2_val) if np.isfinite(h1h2_val) else np.nan,
            "hnr_db": float(hnr_val) if np.isfinite(hnr_val) else np.nan,
            "cpp_db": float(cpp_val) if np.isfinite(cpp_val) else np.nan,
            "rms_db": float(rms),
            "voiced": bool(voiced),
            "sr": int(self.cfg.sr),
            "frame_ms": int(self.cfg.frame_ms),
            "hop_ms": int(self.cfg.hop_ms),
            "pitch_min": float(self.cfg.pitch_min),
            "pitch_max": float(self.cfg.pitch_max),
            "formant_max_hz": int(self.cfg.formant_max_hz),
            "lpc_order": int(self.cfg.lpc_order),
            "formant_method": self.cfg.formant_method,
        }
        self.recorder.push(row)

    def _session_dir(self) -> Path:
        if self.recorder.session_dir is not None:
            return self.recorder.session_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return SESSIONS_DIR / timestamp

    def _read_audio(self, path: str) -> tuple[np.ndarray, int]:
        try:
            data, sr = sf.read(path, always_2d=False)
        except Exception:
            if AudioSegment is None:
                raise
            segment = AudioSegment.from_file(path)
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            if segment.channels > 1:
                samples = samples.reshape(-1, segment.channels).mean(axis=1)
            max_val = float(1 << (8 * segment.sample_width - 1))
            data = samples / max_val
            sr = segment.frame_rate
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), int(sr)

    def _open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open audio file",
            str(Path.home()),
            "Audio Files (*.wav *.flac *.mp3)",
        )
        if path:
            self.load_audio_file(path, target=str(self.source_combo.currentData() or "B"))

    def _export_wav_action(self) -> None:
        path = self.export_wav()
        if path is None:
            QMessageBox.information(self, "Save WAV", "No recording available to save.")

    def _play_action(self) -> None:
        self.play_buffer()

    def _save_session_settings(self) -> None:
        if self.recorder.session_dir is None:
            return
        settings_path = self.recorder.session_dir / "settings.yaml"
        save_preset(settings_path, self.cfg)
