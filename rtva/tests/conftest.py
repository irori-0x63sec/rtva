from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

try:  # pragma: no cover - optional GUI dependency
    from PyQt6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    if QApplication is None:
        pytest.skip("PyQt6 not available")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
