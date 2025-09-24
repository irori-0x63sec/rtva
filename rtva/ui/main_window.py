from __future__ import annotations

import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Deque, Optional

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from rtva.audio import AudioStream, hann
from rtva.config import AnalyzerConfig, save_preset
from rtva.dsp.cepstrum import cpp_db
from rtva.dsp.lpc import burg_formants_parselmouth, lpc_formants
from rtva.dsp.pitch import yin_f0
from rtva.dsp.spectrum import h1_h2_db, stft_mag_db
from rtva.dsp.vad import rms_db, simple_vad
from rtva.logging.recorder import ParquetRecorder
from rtva.ui.controls import ControlsPanel
from rtva.ui.panels import CPPPanel, HarmonicsPanel, PitchPanel, SpectroPanel

try:  # Praat bindings availability
    import parselmouth  # noqa: F401

    PARSELMOUTH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PARSELMOUTH_AVAILABLE = False


SESSIONS_DIR = Path("sessions")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RTVA — Real-Time Voice Analyzer")

        self.cfg = AnalyzerConfig()
        self._cpp_rate_hz = 2.0
        self._cpp_stride = 1
        self._cpp_counter = 0
        self._last_cpp = float("nan")
        self._warn_burg_once = False

        self._parselmouth_available = PARSELMOUTH_AVAILABLE

        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        self.controls = ControlsPanel(self.cfg, parselmouth_available=self._parselmouth_available)
        root_layout.addWidget(self.controls, stretch=0)

        panels = QWidget(self)
        panels_layout = QVBoxLayout()
        panels.setLayout(panels_layout)

        self.pitch = PitchPanel(timespan_sec=10, sr_hop=int(round(1000 / self.cfg.hop_ms)))
        self.spectro = SpectroPanel()
        self.harm = HarmonicsPanel(timespan_sec=10, sr_hop=int(round(1000 / self.cfg.hop_ms)))
        self.cpp = CPPPanel(timespan_sec=30, sr_hop=int(max(1, round(self._cpp_rate_hz))))

        panels_layout.addWidget(self.pitch)
        panels_layout.addWidget(self.spectro)
        panels_layout.addWidget(self.harm)
        panels_layout.addWidget(self.cpp)
        root_layout.addWidget(panels, stretch=1)
        self.setCentralWidget(central)

        self.controls.config_changed.connect(self._apply_config)

        self.stream: Optional[AudioStream] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.recorder = ParquetRecorder(SESSIONS_DIR, rate_hz=self.cfg.log_rate_hz)
        self.recorder.start()

        self._spectrogram_window_sec = 5.0
        self._n_fft = 4096

        self._audio_buffer = np.zeros(1, dtype=np.float32)
        self._analysis_window = np.ones(1, dtype=np.float32)
        self._formant_history: tuple[Deque[float], Deque[float], Deque[float]] = (
            deque(),
            deque(),
            deque(),
        )

        self._apply_config(self.cfg)
        self.timer.start(self.cfg.hop_ms)
        self._save_session_settings()

    def closeEvent(self, event):  # type: ignore[override]
        if self.timer.isActive():
            self.timer.stop()
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
        self.recorder.stop()
        return super().closeEvent(event)

    def _apply_config(self, cfg: AnalyzerConfig) -> None:
        if cfg.formant_method == "burg" and not self._parselmouth_available:
            cfg = replace(cfg, formant_method="lpc")
            self.controls.set_config(cfg)

        self.cfg = cfg
        self._cpp_stride = max(1, int(round((1000.0 / cfg.hop_ms) / self._cpp_rate_hz)))
        self._cpp_counter = 0

        hop_samples = max(1, int(round(cfg.sr * cfg.hop_ms / 1000)))
        frame_samples = max(hop_samples, int(round(cfg.sr * cfg.frame_ms / 1000)))
        self._n_fft = max(2048, 1 << int(np.ceil(np.log2(frame_samples))))

        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
        self.stream = AudioStream(sr=cfg.sr, blocksize=hop_samples)
        self.stream.start()

        buffer_frames = max(1, int(self._spectrogram_window_sec * 1000 / cfg.hop_ms))
        buffer_len = frame_samples + max(0, buffer_frames - 1) * hop_samples
        self._audio_buffer = np.zeros(buffer_len, dtype=np.float32)
        self._analysis_window = hann(frame_samples).astype(np.float32)
        self._formant_history = (
            deque([np.nan] * buffer_frames, maxlen=buffer_frames),
            deque([np.nan] * buffer_frames, maxlen=buffer_frames),
            deque([np.nan] * buffer_frames, maxlen=buffer_frames),
        )

        rate_hz = 1000.0 / cfg.hop_ms
        self.pitch.set_rate(rate_hz)
        self.harm.set_rate(rate_hz)
        self.cpp.set_rate(self._cpp_rate_hz)

        self.timer.setInterval(cfg.hop_ms)
        self.recorder.rate_hz = cfg.log_rate_hz
        self._save_session_settings()

    def _append_audio(self, hop: np.ndarray) -> None:
        hop = hop.astype(np.float32, copy=False)
        n = hop.size
        if n >= self._audio_buffer.size:
            self._audio_buffer[:] = hop[-self._audio_buffer.size :]
        else:
            self._audio_buffer = np.roll(self._audio_buffer, -n)
            self._audio_buffer[-n:] = hop

    def _current_frame(self) -> np.ndarray:
        frame_len = self._analysis_window.size
        return self._audio_buffer[-frame_len:]

    def _tick(self) -> None:
        if self.stream is None:
            return

        try:
            hop = self.stream.read()
        except Exception:
            return

        self._append_audio(hop)
        frame = (self._current_frame() * self._analysis_window).astype(np.float32)

        rms = rms_db(frame)
        voiced = simple_vad(frame, self.cfg.sr)

        f0 = yin_f0(frame, self.cfg.sr, fmin=self.cfg.pitch_min, fmax=self.cfg.pitch_max)
        if not voiced:
            f0 = 0.0
        self.pitch.update_pitch(f0)
        self.pitch.update_stability(self._cents_std_recent())

        f1 = f2 = f3 = np.nan
        if voiced:
            if self.cfg.formant_method == "burg":
                f1, f2, f3 = burg_formants_parselmouth(
                    frame, self.cfg.sr, fmax=self.cfg.formant_max_hz
                )
                if (f1, f2, f3) == (0.0, 0.0, 0.0):
                    if not self._warn_burg_once:
                        print("[WARN] parselmouth unavailable, falling back to LPC formants")
                        self._warn_burg_once = True
                    f1, f2, f3 = lpc_formants(
                        frame,
                        self.cfg.sr,
                        order=self.cfg.lpc_order,
                        fmax=self.cfg.formant_max_hz,
                    )
            else:
                f1, f2, f3 = lpc_formants(
                    frame, self.cfg.sr, order=self.cfg.lpc_order, fmax=self.cfg.formant_max_hz
                )

        formant_vals = [f1, f2, f3]
        for idx, value in enumerate(formant_vals):
            series = self._formant_history[idx]
            series.append(float(value) if value and value > 0 else np.nan)

        h1h2_val = np.nan
        if voiced and f0 > 0:
            h1h2_val = h1_h2_db(frame, self.cfg.sr, f0)
        self.harm.update_value(h1h2_val)

        if voiced and (self._cpp_counter % self._cpp_stride == 0):
            cpp_val = cpp_db(frame, self.cfg.sr)
            self._last_cpp = cpp_val
            self.cpp.update_value(cpp_val)
        elif self._cpp_counter % self._cpp_stride == 0:
            self._last_cpp = float("nan")
            self.cpp.update_value(self._last_cpp)
        self._cpp_counter += 1

        self._update_spectrogram()

        self._push_log(
            f0,
            formant_vals,
            h1h2_val,
            self._last_cpp,
            rms,
            voiced,
        )

    def _update_spectrogram(self) -> None:
        try:
            S_db, freqs, times = stft_mag_db(
                self._audio_buffer,
                self.cfg.sr,
                n_fft=self._n_fft,
                hop=max(1, int(round(self.cfg.sr * self.cfg.hop_ms / 1000))),
                win=self._analysis_window,
            )
        except Exception:
            return

        formants = []
        for series in self._formant_history:
            hist = np.array(series, dtype=float)
            if hist.size < times.size:
                padded = np.full(times.size, np.nan, dtype=float)
                padded[-hist.size :] = hist
                formants.append(padded)
            else:
                formants.append(hist[-times.size :])

        self.spectro.update(S_db, freqs, times, tuple(formants))

    def _cents_std_recent(self, n: int = 50) -> Optional[float]:
        buf = self.pitch.buf.copy()
        vals = buf[~np.isnan(buf)]
        if len(vals) < max(5, n // 2):
            return None
        recent = vals[-n:]
        hz = recent[recent > 0]
        if len(hz) < 5:
            return None
        med = np.median(hz)
        cents = 1200 * np.log2(hz / med)
        return float(np.std(cents))

    def _push_log(
        self,
        f0: float,
        formants: list[float],
        h1h2_val: float,
        cpp_val: float,
        rms: float,
        voiced: bool,
    ) -> None:
        if self.recorder is None:
            return
        row = {
            "timestamp_ns": time.time_ns(),
            "f0_hz": float(f0) if f0 > 0 else np.nan,
            "f1_hz": float(formants[0]) if np.isfinite(formants[0]) and formants[0] > 0 else np.nan,
            "f2_hz": float(formants[1]) if np.isfinite(formants[1]) and formants[1] > 0 else np.nan,
            "f3_hz": float(formants[2]) if np.isfinite(formants[2]) and formants[2] > 0 else np.nan,
            "h1_h2_db": float(h1h2_val) if np.isfinite(h1h2_val) else np.nan,
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

    def _save_session_settings(self) -> None:
        if self.recorder.session_dir is None:
            return
        settings_path = self.recorder.session_dir / "settings.yaml"
        save_preset(settings_path, self.cfg)
