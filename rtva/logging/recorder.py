from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Dict, Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq


class ParquetRecorder:
    def __init__(self, out_dir: Path, rate_hz: int = 10) -> None:
        self.out_dir = Path(out_dir)
        self.rate_hz = max(1, int(rate_hz))
        self._queue: "Queue[Dict[str, Any]]" = Queue()
        self._thread: Optional[Thread] = None
        self._stop = Event()
        self._writer: Optional[pq.ParquetWriter] = None
        self.session_dir: Optional[Path] = None
        self.file_path: Optional[Path] = None
        self._schema = pa.schema(
            [
                ("timestamp_ns", pa.int64()),
                ("f0_hz", pa.float32()),
                ("source", pa.string()),
                ("f1_hz", pa.float32()),
                ("f2_hz", pa.float32()),
                ("f3_hz", pa.float32()),
                ("f4_hz", pa.float32()),
                ("f5_hz", pa.float32()),
                ("h1_h2_db", pa.float32()),
                ("hnr_db", pa.float32()),
                ("cpp_db", pa.float32()),
                ("rms_db", pa.float32()),
                ("voiced", pa.bool_()),
                ("sr", pa.int32()),
                ("frame_ms", pa.int32()),
                ("hop_ms", pa.int32()),
                ("pitch_min", pa.float32()),
                ("pitch_max", pa.float32()),
                ("formant_max_hz", pa.int32()),
                ("lpc_order", pa.int32()),
                ("formant_method", pa.string()),
            ]
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.out_dir / date_str
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.session_dir / "log.parquet"
        self._stop.clear()
        self._thread = Thread(target=self._worker, name="ParquetRecorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._thread = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def push(self, row: Dict[str, Any]) -> None:
        if not self._thread:
            return
        self._queue.put(row)

    def _ensure_writer(self) -> None:
        if self._writer is None:
            if self.file_path is None:
                raise RuntimeError("Recorder has not been started")
            self._writer = pq.ParquetWriter(self.file_path, self._schema)

    def _normalize_rows(self, rows: Iterable[Dict[str, Any]]) -> pa.Table:
        normalized = []
        for row in rows:
            item: Dict[str, Any] = {}
            for field in self._schema.names:
                value = row.get(field)
                if value is None:
                    item[field] = None
                elif field == "voiced":
                    item[field] = bool(value)
                elif field in {"timestamp_ns"}:
                    item[field] = int(value)
                elif self._schema.field(field).type == pa.int32():
                    item[field] = int(value)
                elif self._schema.field(field).type == pa.float32():
                    item[field] = float(value)
                else:
                    item[field] = str(value)
            normalized.append(item)
        return pa.Table.from_pylist(normalized, schema=self._schema)

    def _worker(self) -> None:
        last_flush = time.monotonic()
        pending: list[Dict[str, Any]] = []
        interval = 1.0 / float(max(1, self.rate_hz))
        while not self._stop.is_set() or not self._queue.empty():
            try:
                row = self._queue.get(timeout=0.1)
                pending.append(row)
            except Empty:
                pass

            now = time.monotonic()
            if pending and (now - last_flush >= interval or self._stop.is_set()):
                self._ensure_writer()
                table = self._normalize_rows(pending)
                self._writer.write_table(table)
                pending.clear()
                last_flush = now
                interval = 1.0 / float(max(1, self.rate_hz))

        if pending:
            self._ensure_writer()
            table = self._normalize_rows(pending)
            self._writer.write_table(table)
            pending.clear()
