from pathlib import Path
import logging

from ..simulation.simulation_pipeline import (
    run_relevance_pipeline,
    run_future_timeline_pipeline,
)
from ..news_data import NewsDownloader


def run_pipeline(
    force_future: bool = False,
    use_lazy_retriever: bool = True,
    use_reranker: bool = False,
    luxical_model_path: str | None = None,
    luxical_model_id: str | None = "DatologyAI/luxical-one",
    luxical_model_filename: str | None = "luxical_one_rc4.npz",
) -> None:
    """Run the end-to-end data pipeline."""
    logging.info("Starting run_pipeline")
    news_base_path = Path("/mnt/ssd") / "newstalk-data" / "gdelt-gal"
    news_base_path.mkdir(parents=True, exist_ok=True)
    NewsDownloader(news_base_path).download_latest()

    events_path = Path("data") / "events.jsonl"
    # fetch_polymarkets_events(events_path)

    events_geopol_prob_path = Path("data") / "events_geopol_prob.jsonl"

    # classify_event_geopol_prob(events_geopol_prob_path, events_path)
    # make excel as byproduct
    events_table_path = Path("data") / "events_stats_table.jsonl"

    # generate_event_table(
    #    events_table_path,
    #    events_geopol_prob_path,
    #    events_path
    # )

    # run_present_timeline_pipeline(
    #    active_event_groups_path=Path("data") / "active_event_groups.jsonl",
    #    events_path=events_path,
    #    storage_dir=Path("data") / "simulation",
    #    force_present=False,
    # )

    logging.info(
        "Running relevance pipeline with events_path=%s news_base_path=%s storage_dir=%s",
        events_path,
        news_base_path,
        Path("data") / "simulation",
    )
    logging.info(
        "Relevance module=%s",
        (
            "lazy(agentic)"
            if use_lazy_retriever
            else "eager(reranker)" if use_reranker else "eager(classifier)"
        ),
    )
    relevance_run = run_relevance_pipeline(
        active_event_groups_path=Path("data") / "active_event_groups.jsonl",
        events_path=events_path,
        news_base_path=news_base_path,
        storage_dir=Path("data") / "simulation",
        use_lazy_retriever=use_lazy_retriever,
        luxical_model_path=luxical_model_path,
        luxical_model_id=luxical_model_id,
        luxical_model_filename=luxical_model_filename,
        use_reranker=True,
        reranker_model_name="mixedbread-ai/mxbai-rerank-base-v2",
        # optional:
        reranker_backend="mxbai",
    )
    logging.info("Relevance pipeline completed run_id=%s", relevance_run["run_id"])
    #run_future_timeline_pipeline(
    #    active_event_groups_path=Path("data") / "active_event_groups.jsonl",
    #    events_path=events_path,
    #    storage_dir=Path("data") / "simulation",
    #    relevance_run_id=relevance_run["run_id"],
    #    force_without_relevance=True,
    #)
