from pathlib import Path

from .pipeline import (
    fetch_polymarkets_events,
    generate_event_table,
    classify_event_geopol_prob,
    download_news_past_week,
    download_news_past_day,
)
from ..simulation.simulation_pipeline import (
    run_present_timeline_pipeline,
    run_realtime_simulation_pipeline,
    run_simulation_pipeline,
)

def run_pipeline():
    """Run the end-to-end data pipeline."""
    news_base_path = Path("/mnt/ssd") / "newstalk-data" / "gdelt-gal"
    news_base_path.mkdir(parents=True, exist_ok=True)
    #download_news_past_week(news_base_path)

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

    run_present_timeline_pipeline(
        active_event_groups_path=Path("data") / "active_event_groups.jsonl",
        events_path=events_path,
        storage_dir=Path("data") / "simulation",
        force_present=False,
    )

    run_realtime_simulation_pipeline(
        active_event_groups_path=Path("data") / "active_event_groups.jsonl",
        events_path=events_path,
        news_base_path=news_base_path,
        storage_dir=Path("data") / "simulation",
    )
