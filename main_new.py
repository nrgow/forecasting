import json
import logging
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.processing.run_pipeline import run_pipeline
from src.simulation.openforecaster import run_openforecaster_pipeline
from src.simulation.simulation_pipeline import build_future_timeline_llm_inputs


class CLI:
    def run_pipeline(
        self,
        force_future: bool = False,
        use_lazy_retriever: bool = False,
        use_reranker: bool = True,
    ) -> None:
        """Run the main data pipeline."""
        run_pipeline(
            force_future=force_future,
            use_lazy_retriever=use_lazy_retriever,
            use_reranker=use_reranker,
        )

    def run_openforecaster(
        self,
        max_news: int = 8,
        max_new_tokens: int = 512 * 4,
        temperature: float = 0.2,
        model_name: str = "nikhilchandak/OpenForecaster-8B",
    ) -> None:
        """Generate OpenForecaster analyses for active event groups."""
        run_openforecaster_pipeline(
            active_event_groups_path=Path("data") / "active_event_groups.jsonl",
            events_path=Path("data") / "events.jsonl",
            storage_dir=Path("data") / "simulation",
            max_news=max_news,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_name=model_name,
        )

    def dump_future_timeline_inputs(
        self,
        active_event_groups_path: Path = Path("data") / "active_event_groups.jsonl",
        events_path: Path = Path("data") / "events.jsonl",
        storage_dir: Path = Path("data") / "simulation",
        relevance_run_id: str | None = None,
        force_without_relevance: bool = False,
        max_recent_items: int = 50,
        recent_bucket_count: int = 8,
    ) -> None:
        """Print JSONL inputs for future timeline LLM generation."""
        records = build_future_timeline_llm_inputs(
            active_event_groups_path=active_event_groups_path,
            events_path=events_path,
            storage_dir=storage_dir,
            relevance_run_id=relevance_run_id,
            force_without_relevance=force_without_relevance,
            max_recent_items=max_recent_items,
            recent_bucket_count=recent_bucket_count,
        )
        for record in records:
            print(json.dumps(record, ensure_ascii=True))


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    )
    fire.Fire(CLI)
