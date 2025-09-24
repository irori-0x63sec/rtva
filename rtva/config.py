from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict

try:  # Prefer PyYAML if available
    import yaml
except Exception:  # pragma: no cover - YAML library may be missing
    yaml = None  # type: ignore


@dataclass
class AnalyzerConfig:
    sr: int = 48000
    frame_ms: int = 32
    hop_ms: int = 10
    pitch_min: float = 120.0
    pitch_max: float = 400.0
    formant_method: str = "burg"
    formant_max_hz: int = 5500
    lpc_order: int = 16
    log_rate_hz: int = 10


def _coerce(field_name: str, value: Any) -> Any:
    for f in fields(AnalyzerConfig):
        if f.name == field_name:
            typ = f.type
            if typ is int:
                return int(value)
            if typ is float:
                return float(value)
            if typ is str:
                return str(value)
            return value
    return value


def save_preset(path: str | Path, cfg: AnalyzerConfig) -> None:
    data = asdict(cfg)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if yaml is not None:
        with file_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
        return

    with file_path.open("w", encoding="utf-8") as fh:
        for key, value in data.items():
            if isinstance(value, str):
                fh.write(f"{key}: '{value}'\n")
            else:
                fh.write(f"{key}: {value}\n")


def load_preset(path: str | Path) -> AnalyzerConfig:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    if yaml is not None:
        with file_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data: Dict[str, Any] = {}
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, raw = line.partition(":")
                if not _:
                    continue
                key = key.strip()
                raw = raw.strip()
                if raw.startswith("'") and raw.endswith("'"):
                    value: Any = raw[1:-1]
                else:
                    try:
                        value = int(raw)
                    except ValueError:
                        try:
                            value = float(raw)
                        except ValueError:
                            if raw.lower() in {"true", "false"}:
                                value = raw.lower() == "true"
                            else:
                                value = raw
                data[key] = value

    coerced = {name: _coerce(name, value) for name, value in data.items()}
    return AnalyzerConfig(**coerced)
