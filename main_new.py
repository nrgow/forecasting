import datetime as dt
import json
import logging
import time
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.processing.run_pipeline import run_pipeline
from src.portfolio_optimize import (
    fetch_latest_market_prices,
    load_run_market_probabilities,
    run_portfolio_optimizer,
    run_portfolio_optimizer_from_prices,
)
from src.simulation.openforecaster import run_openforecaster_pipeline
from src.simulation.storage import SimulationStorage
from src.simulation.simulation_pipeline import (
    BoilerplateDeleter,
    build_future_timeline_llm_inputs,
    build_future_timeline_llm_inputs_perplexity,
    iter_news_articles,
    iter_news_paths,
    run_future_timeline_perplexity_pipeline,
    run_present_timeline_perplexity_pipeline,
)


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
        max_recent_items: int = 200,
        recent_bucket_count: int = 10,
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

    def dump_future_timeline_inputs_perplexity(
        self,
        active_event_groups_path: Path = Path("data") / "active_event_groups.jsonl",
        events_path: Path = Path("data") / "events.jsonl",
        storage_dir: Path = Path("data") / "simulation",
    ) -> None:
        """Print JSONL inputs using Perplexity present timelines only."""
        records = build_future_timeline_llm_inputs_perplexity(
            active_event_groups_path=active_event_groups_path,
            events_path=events_path,
            storage_dir=storage_dir,
        )
        for record in records:
            print(json.dumps(record, ensure_ascii=True))

    def run_present_timeline_perplexity(
        self,
        active_event_groups_path: Path = Path("data") / "active_event_groups.jsonl",
        events_path: Path = Path("data") / "events.jsonl",
        storage_dir: Path = Path("data") / "simulation",
        model: str = "perplexity/sonar-deep-research",
        temperature: float = 0.2,
        max_tokens: int = 12000,
        current_date: str | None = None,
        force_present: bool = False,
    ) -> None:
        """Generate Perplexity present timelines for active event groups."""
        run_present_timeline_perplexity_pipeline(
            active_event_groups_path=active_event_groups_path,
            events_path=events_path,
            storage_dir=storage_dir,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            current_date=current_date,
            force_present=force_present,
        )

    def run_future_timeline_perplexity(
        self,
        active_event_groups_path: Path = Path("data") / "active_event_groups.jsonl",
        events_path: Path = Path("data") / "events.jsonl",
        storage_dir: Path = Path("data") / "simulation",
        timeline_models: list[str] | None = [
            "openrouter/x-ai/grok-4.1-fast",
            "openrouter/google/gemini-3-flash-preview",
        ],
        timeline_temps: list[float] | None = [0.7],
        timeline_rollouts: int | None = 5,
    ) -> None:
        """Generate future timelines from Perplexity present timelines only."""
        run_future_timeline_perplexity_pipeline(
            active_event_groups_path=active_event_groups_path,
            events_path=events_path,
            storage_dir=storage_dir,
            timeline_models=timeline_models,
            timeline_temps=timeline_temps,
            timeline_rollouts=timeline_rollouts,
        )

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

    def optimize_perplexity_run_portfolio(
        self,
        max_fraction_per_market: float,
        max_total_fraction: float,
        kelly_fraction: float = 1.0,
        turnover_penalty: float = 0.0,
        epsilon: float = 1e-9,
        solver: str = "ECOS",
        round_threshold: float = 1e-6,
        storage_dir: Path = Path("data") / "simulation",
        probabilities_path: Path = Path("data")
        / "simulation"
        / "estimated_event_probabilities.jsonl",
        run_id: str | None = None,
    ) -> None:
        """Optimize portfolio from the latest Perplexity future timeline run."""
        overall_started_at = time.perf_counter()
        logging.info("Perplexity portfolio optimization starting")
        storage = SimulationStorage(storage_dir)
        if run_id is None:
            logging.info("Resolving latest Perplexity run id from %s", storage.runs_path)
            with storage.runs_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record["present_timeline_source"] != "perplexity":
                        continue
                    run_id = record["run_id"]
        if run_id is None:
            raise ValueError("No Perplexity future timeline runs found.")
        logging.info("Using run_id=%s", run_id)
        probability_records = load_run_market_probabilities(probabilities_path, run_id)
        market_ids = {record["market_id"] for record in probability_records}
        market_prices = fetch_latest_market_prices(market_ids)
        market_price_rows = []
        for record in probability_records:
            market_id = record["market_id"]
            market_price_rows.append(
                {
                    "event_group_id": record["event_group_id"],
                    "market_id": market_id,
                    "market_question": record["market_question"],
                    "market_slug": record["market_slug"],
                    "p_yes": market_prices[market_id],
                }
            )
        optimizer_output = run_portfolio_optimizer_from_prices(
            probability_records=probability_records,
            market_prices=market_prices,
            max_fraction_per_market=max_fraction_per_market,
            max_total_fraction=max_total_fraction,
            kelly_fraction=kelly_fraction,
            turnover_penalty=turnover_penalty,
            epsilon=epsilon,
            solver=solver,
            round_threshold=round_threshold,
        )
        elapsed = time.perf_counter() - overall_started_at
        logging.info(
            "Perplexity portfolio optimization completed in %.2fs", elapsed
        )
        output = {
            "run_id": run_id,
            "probabilities": probability_records,
            "market_prices": market_price_rows,
            "portfolio": optimizer_output,
        }
        print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    )
    fire.Fire(CLI)
