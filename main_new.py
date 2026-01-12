import logging
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.processing.run_pipeline import run_pipeline
from src.simulation.openforecaster import run_openforecaster_pipeline


class CLI:
    def run_pipeline(
        self,
        force_future: bool = False,
        use_lazy_retriever: bool = True,
    ) -> None:
        """Run the main data pipeline."""
        run_pipeline(force_future=force_future, use_lazy_retriever=use_lazy_retriever)

    def run_openforecaster(
        self,
        max_news: int = 8,
        max_new_tokens: int = 512*4,
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


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    )
    fire.Fire(CLI)
