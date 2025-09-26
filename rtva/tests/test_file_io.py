from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("PyQt6.QtWidgets")

from rtva.ui.main_window import MainWindow


def test_load_audio_file_populates_layer(tmp_path: Path, qapp) -> None:  # noqa: ARG001
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(wav_path, data, sr)

    window = MainWindow()
    try:
        window.load_audio_file(str(wav_path), target="B")
        state = window._sources["B"]
        assert state.recording
        layer = window.spectro._last_layers.get("B")
        assert layer is not None
        S_db, _, _, formants = layer
        assert S_db.size > 0
        assert any(np.isfinite(f[-1]) for f in formants)
    finally:
        window.close()
