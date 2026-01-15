import datetime as dt
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeepSearchTraceStore:
    """Write deep search agent traces to JSONL files by event group."""

    base_dir: Path

    def __post_init__(self) -> None:
        """Ensure the trace directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def append_trace(self, event_group_id: str, record: dict) -> Path:
        """Append a trace record for the event group and return the file path."""
        path = self.base_dir / f"{event_group_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
        return path


def trace_timestamp() -> str:
    """Return an ISO8601 timestamp for trace records."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log_trace_write_started(event_group_id: str, path: Path) -> float:
    """Log the start of a trace write and return the timer start."""
    logging.info(
        "Deep search trace write starting event_group_id=%s path=%s",
        event_group_id,
        path,
    )
    return time.perf_counter()


def log_trace_write_completed(
    event_group_id: str, path: Path, started_at: float
) -> None:
    """Log completion for a trace write."""
    elapsed = time.perf_counter() - started_at
    logging.info(
        "Deep search trace write completed event_group_id=%s path=%s seconds=%.2f",
        event_group_id,
        path,
        elapsed,
    )
