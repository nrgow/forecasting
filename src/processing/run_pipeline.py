from pathlib import Path

from .pipeline import fetch_polymarkets_events, generate_event_table, classify_event_geopol_prob


def run_pipeline():
    events_path = Path("data") / "events.jsonl"
    fetch_polymarkets_events(events_path)
    
    events_geopol_prob_path = Path("data") / "events_geopol_prob.jsonl"
    classify_event_geopol_prob(events_geopol_prob_path, events_path)
    # make excel as byproduct
    events_table_path = Path("data") / f'events_stats_table.jsonl'
    generate_event_table(
        events_table_path,
        events_geopol_prob_path,
        events_path
    )
