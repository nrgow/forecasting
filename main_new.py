import datetime as dt
import json
import logging
import time
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.processing.run_pipeline import run_pipeline
from src.portfolio_optimize import run_portfolio_optimizer
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

    def optimize_portfolio(
        self,
        max_fraction_per_market: float,
        max_total_fraction: float,
        kelly_fraction: float = 1.0,
        turnover_penalty: float = 0.0,
        epsilon: float = 1e-9,
        solver: str = "ECOS",
        round_threshold: float = 1e-6,
        probabilities_path: Path = Path("data")
        / "simulation"
        / "estimated_event_probabilities.jsonl",
        events_path: Path = Path("data") / "events.jsonl",
    ) -> None:
        """Optimize the portfolio using latest probabilities and market prices."""
        logging.info("Portfolio optimization entrypoint starting")
        output = run_portfolio_optimizer(
            probabilities_path=probabilities_path,
            events_path=events_path,
            max_fraction_per_market=max_fraction_per_market,
            max_total_fraction=max_total_fraction,
            kelly_fraction=kelly_fraction,
            turnover_penalty=turnover_penalty,
            epsilon=epsilon,
            solver=solver,
            round_threshold=round_threshold,
        )
        logging.info("Portfolio optimization entrypoint finished")
        print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    )
    fire.Fire(CLI)
