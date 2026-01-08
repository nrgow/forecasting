import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SimulationStorage:
    """Filesystem-backed storage for simulation artifacts."""

    base_dir: Path

    def __post_init__(self) -> None:
        """Ensure storage directories and paths are initialized."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.present_timelines_path = self.base_dir / "present_timelines.jsonl"
        self.future_timelines_path = self.base_dir / "future_timelines.jsonl"
        self.probabilities_path = self.base_dir / "estimated_event_probabilities.jsonl"
        self.relevance_path = self.base_dir / "realtime_relevance.jsonl"
        self.runs_path = self.base_dir / "simulation_runs.jsonl"

    def append_present_timeline(self, record: dict) -> None:
        """Append a present timeline record."""
        self._append_jsonl(self.present_timelines_path, record)

    def append_future_timeline(self, record: dict) -> None:
        """Append a future timeline record."""
        self._append_jsonl(self.future_timelines_path, record)

    def append_probability_estimate(self, record: dict) -> None:
        """Append an event probability estimate record."""
        self._append_jsonl(self.probabilities_path, record)

    def append_relevance_judgment(self, record: dict) -> None:
        """Append a relevance judgment record."""
        self._append_jsonl(self.relevance_path, record)

    def append_run_metadata(self, record: dict) -> None:
        """Append a simulation run metadata record."""
        self._append_jsonl(self.runs_path, record)

    def load_present_timeline_index(self) -> dict[str, dict]:
        """Return a mapping of event_group_id to the latest present timeline record."""
        records = self._iter_jsonl(self.present_timelines_path)
        return {record["event_group_id"]: record for record in records}

    def load_relevance_index(self) -> set[tuple[str, str]]:
        """Return a set of (event_group_id, article_id) for existing judgments."""
        records = self._iter_jsonl(self.relevance_path)
        return {(record["event_group_id"], record["article_id"]) for record in records}

    def last_run_metadata(self) -> dict | None:
        """Return the last run metadata record if present."""
        last = None
        for record in self._iter_jsonl(self.runs_path):
            last = record
        return last

    def _iter_jsonl(self, path: Path) -> Iterable[dict]:
        """Yield JSONL records from a file if it exists."""
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)

    def _append_jsonl(self, path: Path, record: dict) -> None:
        """Append a single JSONL record."""
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
