from pathlib import Path

from ..news_data import NewsDownloader
from .pipeline import fetch_polymarkets_events, generate_event_table, classify_event_geopol_prob, download_news_past_week


def run_pipeline():
    news_base_path = Path("/mnt/ssd") / "newstalk-data" / "gdelt-gal"
    news_base_path.mkdir(parents=True, exist_ok=True)
    download_news_past_week(news_base_path)

    events_path = Path("data") / "events.jsonl"
    #fetch_polymarkets_events(events_path)
    
    events_geopol_prob_path = Path("data") / "events_geopol_prob.jsonl"
    
    #classify_event_geopol_prob(events_geopol_prob_path, events_path)
    # make excel as byproduct
    events_table_path = Path("data") / f'events_stats_table.jsonl'
    generate_event_table(
        events_table_path,
        events_geopol_prob_path,
        events_path
    )
