from pathlib import Path
import logging

from .pipeline import (
    fetch_polymarkets_events,
    generate_event_table,
    classify_event_geopol_prob,
    download_news_past_week,
)
from ..news_data import NewsDownloader
from ..simulation.simulation_pipeline import (
    run_present_timeline_pipeline,
    run_relevance_pipeline,
    run_future_timeline_pipeline,
)

def run_pipeline(force_future: bool = False) -> None:
    """Run the end-to-end data pipeline."""
    logging.info("Starting run_pipeline")
    news_base_path = Path("/mnt/ssd") / "newstalk-data" / "gdelt-gal"
    news_base_path.mkdir(parents=True, exist_ok=True)
    NewsDownloader(news_base_path).download_latest()

    events_path = Path("data") / "events.jsonl"
    #fetch_polymarkets_events(events_path)
    
    events_geopol_prob_path = Path("data") / "events_geopol_prob.jsonl"
    
    #classify_event_geopol_prob(events_geopol_prob_path, events_path)
    # make excel as byproduct
    events_table_path = Path("data") / f'events_stats_table.jsonl'

    #generate_event_table(
    #    events_table_path,
    #    events_geopol_prob_path,
    #    events_path
    #)

    #run_present_timeline_pipeline(
    #    active_event_groups_path=Path("data") / "active_event_groups.jsonl",
    #    events_path=events_path,
    #    storage_dir=Path("data") / "simulation",
    #    force_present=False,
    #)

    logging.info(
        "Running relevance pipeline with events_path=%s news_base_path=%s storage_dir=%s",
        events_path,
        news_base_path,
        Path("data") / "simulation",
    )
    relevance_run = run_relevance_pipeline(
        active_event_groups_path=Path("data") / "active_event_groups.jsonl",
        events_path=events_path,
        news_base_path=news_base_path,
        storage_dir=Path("data") / "simulation",
    )
    logging.info("Relevance pipeline completed run_id=%s", relevance_run["run_id"])
    run_future_timeline_pipeline(
        active_event_groups_path=Path("data") / "active_event_groups.jsonl",
        events_path=events_path,
        storage_dir=Path("data") / "simulation",
        relevance_run_id=relevance_run["run_id"],
        force_without_relevance=True,
    )
