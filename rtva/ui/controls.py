from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rtva.config import AnalyzerConfig, load_preset, save_preset


class ControlsPanel(QWidget):
    config_changed = pyqtSignal(AnalyzerConfig)

    def __init__(self, cfg: AnalyzerConfig, parselmouth_available: bool = True) -> None:
        super().__init__()
        self._cfg = cfg
        self._parselmouth_available = parselmouth_available
        self._updating = False

        layout = QVBoxLayout()
        self.setLayout(layout)

        form_box = QGroupBox("Analyzer")
        form_layout = QFormLayout()
        form_box.setLayout(form_layout)

        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 192000)
        self.sr_spin.setSingleStep(1000)
        form_layout.addRow("Sample rate", self.sr_spin)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(8, 128)
        self.frame_spin.setSingleStep(2)
        form_layout.addRow("Frame (ms)", self.frame_spin)

        self.hop_spin = QSpinBox()
        self.hop_spin.setRange(1, 100)
        form_layout.addRow("Hop (ms)", self.hop_spin)

        self.pitch_min_spin = QDoubleSpinBox()
        self.pitch_min_spin.setRange(20.0, 1000.0)
        self.pitch_min_spin.setDecimals(1)
        self.pitch_min_spin.setSingleStep(5.0)
        form_layout.addRow("Pitch min", self.pitch_min_spin)

        self.pitch_max_spin = QDoubleSpinBox()
        self.pitch_max_spin.setRange(50.0, 2000.0)
        self.pitch_max_spin.setDecimals(1)
        self.pitch_max_spin.setSingleStep(5.0)
        form_layout.addRow("Pitch max", self.pitch_max_spin)

        self.formant_combo = QComboBox()
        self.formant_combo.addItems(["burg", "lpc"])
        if not parselmouth_available:
            index = self.formant_combo.findText("burg")
            if index >= 0:
                self.formant_combo.removeItem(index)
        form_layout.addRow("Formant method", self.formant_combo)

        self.formant_max_spin = QSpinBox()
        self.formant_max_spin.setRange(1000, 10000)
        self.formant_max_spin.setSingleStep(100)
        form_layout.addRow("Formant max (Hz)", self.formant_max_spin)

        self.lpc_order_spin = QSpinBox()
        self.lpc_order_spin.setRange(8, 32)
        form_layout.addRow("LPC order", self.lpc_order_spin)

        self.log_rate_spin = QSpinBox()
        self.log_rate_spin.setRange(1, 60)
        form_layout.addRow("Log rate (Hz)", self.log_rate_spin)

        layout.addWidget(form_box)

        buttons_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save preset")
        self.save_btn.clicked.connect(self._save_preset)
        buttons_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load preset")
        self.load_btn.clicked.connect(self._load_preset)
        buttons_layout.addWidget(self.load_btn)
        layout.addLayout(buttons_layout)

        layout.addStretch(1)

        self._connect_signals()
        self.set_config(cfg)

    def _connect_signals(self) -> None:
        widgets = [
            self.sr_spin,
            self.frame_spin,
            self.hop_spin,
            self.pitch_min_spin,
            self.pitch_max_spin,
            self.formant_combo,
            self.formant_max_spin,
            self.lpc_order_spin,
            self.log_rate_spin,
        ]
        for widget in widgets:
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._emit_config)
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(self._emit_config)
            else:
                widget.valueChanged.connect(self._emit_config)

    def set_config(self, cfg: AnalyzerConfig) -> None:
        self._updating = True
        self._cfg = cfg
        try:
            self.sr_spin.setValue(cfg.sr)
            self.frame_spin.setValue(cfg.frame_ms)
            self.hop_spin.setValue(cfg.hop_ms)
            self.pitch_min_spin.setValue(cfg.pitch_min)
            self.pitch_max_spin.setValue(cfg.pitch_max)
            idx = self.formant_combo.findText(cfg.formant_method)
            if idx >= 0:
                self.formant_combo.setCurrentIndex(idx)
            else:
                self.formant_combo.setCurrentIndex(0)
            self.formant_max_spin.setValue(cfg.formant_max_hz)
            self.lpc_order_spin.setValue(cfg.lpc_order)
            self.log_rate_spin.setValue(cfg.log_rate_hz)
        finally:
            self._updating = False

    def _emit_config(self) -> None:
        if self._updating:
            return
        cfg = AnalyzerConfig(
            sr=int(self.sr_spin.value()),
            frame_ms=int(self.frame_spin.value()),
            hop_ms=int(self.hop_spin.value()),
            pitch_min=float(self.pitch_min_spin.value()),
            pitch_max=float(self.pitch_max_spin.value()),
            formant_method=self.formant_combo.currentText(),
            formant_max_hz=int(self.formant_max_spin.value()),
            lpc_order=int(self.lpc_order_spin.value()),
            log_rate_hz=int(self.log_rate_spin.value()),
        )
        if not self._parselmouth_available and cfg.formant_method == "burg":
            cfg.formant_method = "lpc"
            idx = self.formant_combo.findText("lpc")
            if idx >= 0:
                self.formant_combo.setCurrentIndex(idx)
        self._cfg = cfg
        self.config_changed.emit(cfg)

    def _save_preset(self) -> None:
        directory = self._default_dir()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save preset",
            str(directory / "preset.yaml"),
            "YAML Files (*.yaml)",
        )
        if path:
            save_preset(path, self._cfg)

    def _load_preset(self) -> None:
        directory = self._default_dir()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load preset",
            str(directory),
            "YAML Files (*.yaml)",
        )
        if not path:
            return
        cfg = load_preset(path)
        self.set_config(cfg)
        self.config_changed.emit(cfg)

    def _default_dir(self) -> Path:
        base = Path.home() / "rtva_presets"
        base.mkdir(parents=True, exist_ok=True)
        return base
