from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets")

from rtva.ui.main_window import MainWindow


def test_export_creates_files(tmp_path: Path, qapp) -> None:  # noqa: ARG001
    window = MainWindow()
    try:
        window.recorder.session_dir = tmp_path
        window._sources["A"].recording = [np.ones(4800, dtype=np.float32)]
        wav_path = window.export_wav(target="A")
        assert wav_path is not None
        assert wav_path.exists()
        assert wav_path.stat().st_size > 0

        png_path = window.export_snapshot_png()
        assert png_path is not None
        assert png_path.exists()
        assert png_path.stat().st_size > 0
    finally:
        window.close()
