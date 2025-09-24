# RTVA — Real-Time Voice Analyzer (Praat-based)

リアルタイムで F0, Formant, H1–H2, CPP を可視化・記録するツール。

## Quick Start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -e .
python -m rtva.app
```

## Features (MVP)
- Mic入力 → F0(YIN)・スペクトログラム・簡易Formant(LPC)
- 4パネルGUI（PyQt6+pyqtgraph）
- Parquetログ保存

## Roadmap
- H1–H2, CPPの安定化
- Praat(parselmouth)比較モード
- プリセット保存、スナップショットPNG
