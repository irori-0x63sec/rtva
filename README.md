# RTVA — Real-Time Voice Analyzer (Praat-based)

リアルタイムで F0, Formant, H1–H2, CPP を可視化・記録するツール。

## Quick Start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -e .
python -m rtva.app
```
 
## Setup

### Linux (Ubuntu/Debian)
1. Install system dependencies:
   ```bash
   sudo apt-get update && sudo apt-get install -y libsndfile1
   ```
2. Create and activate a virtual environment, then install RTVA with dev extras:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Run the test suite:
   ```bash
   pytest -q -ra
   ```

### macOS
1. Install [Homebrew](https://brew.sh/) if not already available, then install libsndfile:
   ```bash
   brew install libsndfile
   ```
2. Create/activate a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Run the test suite:
   ```bash
   pytest -q -ra
   ```

### Windows
1. Create and activate a virtual environment (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```
2. Install dependencies (prebuilt wheels of `soundfile` include libsndfile):
   ```powershell
   pip install -e ".[dev]"
   ```
3. Run the test suite:
   ```powershell
   pytest -q -ra
   ```

## Features (MVP)
- Mic入力 → F0(YIN)・スペクトログラム・簡易Formant(LPC)
- 4パネルGUI（PyQt6+pyqtgraph）
- Parquetログ保存

## Roadmap
- H1–H2, CPPの安定化
- Praat(parselmouth)比較モード
- プリセット保存、スナップショットPNG
