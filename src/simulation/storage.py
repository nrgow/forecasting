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
        self.zero_shot_path = self.base_dir / "news_zero_shot.jsonl"
        self.openforecaster_path = self.base_dir / "openforecaster_analyses.jsonl"
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

    def append_zero_shot_record(self, record: dict) -> None:
        """Append a zero-shot classification record."""
        self._append_jsonl(self.zero_shot_path, record)

    def append_openforecaster_analysis(self, record: dict) -> None:
        """Append an OpenForecaster analysis record."""
        self._append_jsonl(self.openforecaster_path, record)

    def append_run_metadata(self, record: dict) -> None:
        """Append a simulation run metadata record."""
        self._append_jsonl(self.runs_path, record)

    def load_present_timeline_index(self) -> dict[str, dict]:
        """Return a mapping of event_group_id to the latest present timeline record."""
        records = self._iter_jsonl(self.present_timelines_path)
        return {record["event_group_id"]: record for record in records}

    def purge_present_timelines(self, event_group_id: str) -> int:
        """Remove present timeline records for the provided event group id."""
        if not self.present_timelines_path.exists():
            return 0
        kept: list[dict] = []
        removed = 0
        with self.present_timelines_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["event_group_id"] == event_group_id:
                    removed += 1
                    continue
                kept.append(record)
        temp_path = self.present_timelines_path.with_suffix(".jsonl.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        temp_path.replace(self.present_timelines_path)
        return removed

    def load_relevance_index(self) -> set[tuple[str, str]]:
        """Return a set of (event_group_id, article_id) for existing judgments."""
        records = self._iter_jsonl(self.relevance_path)
        return {(record["event_group_id"], record["article_id"]) for record in records}

    def load_relevance_records(self, run_id: str) -> list[dict]:
        """Return relevance judgment records for a specific run."""
        records = self._iter_jsonl(self.relevance_path)
        return [record for record in records if record["run_id"] == run_id]

    def load_latest_relevance_records_for_group(self, event_group_id: str) -> list[dict]:
        """Return relevance records for the latest run of an event group."""
        latest_run_id = None
        latest_records: list[dict] = []
        for record in self._iter_jsonl(self.relevance_path):
            if record["event_group_id"] != event_group_id:
                continue
            if latest_run_id != record["run_id"]:
                latest_run_id = record["run_id"]
                latest_records = []
            latest_records.append(record)
        return latest_records

    def load_latest_relevant_news(self, event_group_id: str) -> list[dict]:
        """Return relevant news records from the latest relevance run."""
        records = self.load_latest_relevance_records_for_group(event_group_id)
        return [record for record in records if record["relevant"]]

    def last_relevance_run_id(self) -> str | None:
        """Return the most recent relevance run id if present."""
        last = None
        for record in self._iter_jsonl(self.relevance_path):
            last = record["run_id"]
        return last

    def last_relevance_judged_at(self) -> str | None:
        """Return the most recent relevance judgment timestamp if present."""
        last = None
        for record in self._iter_jsonl(self.relevance_path):
            last = record["judged_at"]
        return last

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
