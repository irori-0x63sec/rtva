from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    from typing import get_args, get_origin
except ImportError:  # pragma: no cover - Python <3.8 fallback

    def get_origin(tp):  # type: ignore
        return getattr(tp, "__origin__", None)

    def get_args(tp):  # type: ignore
        return getattr(tp, "__args__", ())


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
    formant_method: str = "auto"
    formant_max_hz: int = 5500
    lpc_order: int = 16
    log_rate_hz: int = 10
    formant_count: int = 5
    colormap: str = "magma"
    db_min: float | None = None
    db_max: float | None = None
    show_formants: tuple[bool, bool, bool, bool, bool] = (
        True,
        True,
        True,
        True,
        True,
    )


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
            if typ is bool:
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes", "on"}
                return bool(value)
            origin = get_origin(typ)
            if origin in (tuple, list):
                args: Iterable[Any] = get_args(typ)
                if isinstance(value, (list, tuple)):
                    if all(arg is bool for arg in args):
                        seq = [bool(v) for v in value]
                        if origin is tuple:
                            length = len(tuple(args)) or len(seq)
                            return tuple(seq[:length])
                        return seq
                return value
            if origin is not None and any(arg is type(None) for arg in get_args(typ)):
                for arg in get_args(typ):
                    if arg is type(None) and (value in (None, "null", "None")):
                        return None
                    if arg is float:
                        return float(value) if value is not None else None
                    if arg is int:
                        return int(value) if value is not None else None
                    if arg is str:
                        return str(value) if value is not None else None
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
