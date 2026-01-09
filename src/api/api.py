from __future__ import annotations

import json
from contextlib import asynccontextmanager
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ..prediction_markets import Events, EventGroup


def load_active_event_groups() -> set[str]:
    """Load active event group ids from JSONL."""
    data_path = Path(__file__).resolve().parents[2] / "data" / "active_event_groups.jsonl"
    lines = data_path.read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["event_group_id"] for line in lines}


def load_events_stats_table(active_event_groups: set[str]) -> list[dict]:
    """Load the events stats table from JSONL and mark active groups."""
    data_path = Path(__file__).resolve().parents[2] / "data" / "events_stats_table.jsonl"
    lines = data_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    for record in records:
        record["active"] = record["event_group_id"] in active_event_groups
    return records


def load_event_group_index() -> dict[str, EventGroup]:
    """Load EventGroups from JSONL and index by event_group_id."""
    data_path = Path(__file__).resolve().parents[2] / "data" / "events.jsonl"
    events = Events.from_file(data_path)
    return {group.id(): group for group in events.group_by_title_template()}


def load_jsonl_records(path: Path) -> list[dict]:
    """Load JSONL records from a file if it exists."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def index_future_timelines(records: list[dict]) -> dict[str, list[dict]]:
    """Index future timeline records by event_group_id."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["event_group_id"]].append(record)
    return grouped


def index_relevance(records: list[dict]) -> dict[str, list[dict]]:
    """Index relevant news records by event_group_id."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["relevant"]:
            grouped[record["event_group_id"]].append(record)
    return grouped


def index_latest_market_probabilities(records: list[dict]) -> dict[str, list[dict]]:
    """Index the latest market probabilities per event group."""
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        event_group_id = record["event_group_id"]
        market_id = record["market_id"]
        if market_id not in grouped[event_group_id]:
            grouped[event_group_id][market_id] = record
            continue
        if record["generated_at"] > grouped[event_group_id][market_id]["generated_at"]:
            grouped[event_group_id][market_id] = record
    return {
        event_group_id: sorted(
            markets.values(), key=lambda item: item["generated_at"], reverse=True
        )
        for event_group_id, markets in grouped.items()
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the events stats table once at startup."""
    active_event_groups = load_active_event_groups()
    app.state.active_event_groups = active_event_groups
    app.state.events_stats_table = load_events_stats_table(active_event_groups)
    app.state.event_group_index = load_event_group_index()
    simulation_dir = Path(__file__).resolve().parents[2] / "data" / "simulation"
    app.state.present_timelines = {
        record["event_group_id"]: record
        for record in load_jsonl_records(simulation_dir / "present_timelines.jsonl")
    }
    app.state.future_timelines = index_future_timelines(
        load_jsonl_records(simulation_dir / "future_timelines.jsonl")
    )
    app.state.relevance = index_relevance(
        load_jsonl_records(simulation_dir / "realtime_relevance.jsonl")
    )
    app.state.market_probabilities = index_latest_market_probabilities(
        load_jsonl_records(simulation_dir / "estimated_event_probabilities.jsonl")
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/event_stats_table")
def event_stats_table() -> list[dict]:
    """Return the full events stats table."""
    return app.state.events_stats_table


@app.get("/event_groups/{event_group_id}")
def event_group_detail(event_group_id: str) -> dict:
    """Return detailed data for a single event group."""
    event_group = app.state.event_group_index.get(event_group_id)
    if event_group is None:
        raise HTTPException(status_code=404, detail="Event group not found")
    open_markets = [
        {
            "id": market.id,
            "question": market.question,
            "slug": market.slug,
            "liquidity": market.liquidity,
            "volume": market.volume,
            "end_date": market.end_date,
            "active": market.active,
            "closed": market.closed,
            "outcomes": market.outcomes,
            "outcome_prices": market.outcome_prices,
        }
        for market in event_group.open_markets()
        if market.active and not market.closed
    ]
    relevant_news = sorted(
        app.state.relevance[event_group_id],
        key=itemgetter("article_published_at"),
        reverse=True,
    )
    present_record = app.state.present_timelines.get(event_group_id)
    present_timeline = None
    if present_record is not None:
        merged = present_record["timeline"]["merged"]
        present_timeline = {
            "generated_at": present_record["generated_at"],
            "summary": merged["merged_timeline"],
            "keyterms": present_record["derived_keyterms"],
        }
    future_timelines = [
        {
            "generated_at": record["generated_at"],
            "scenario": record["scenario"],
            "results": record["results"],
        }
        for record in app.state.future_timelines[event_group_id]
    ]
    if event_group_id in app.state.market_probabilities:
        market_probabilities = app.state.market_probabilities[event_group_id]
    else:
        market_probabilities = []
    events = [
        {
            "id": event.id,
            "title": event.title,
            "description": event.description,
            "start_date": event.start_date,
            "end_date": event.end_date,
            "url": event.url(),
        }
        for event in event_group.events
    ]
    return {
        "event_group_id": event_group_id,
        "event_group_title": event_group.template_title,
        "active": event_group_id in app.state.active_event_groups,
        "events": events,
        "open_markets": open_markets,
        "market_probabilities": market_probabilities,
        "relevant_news": relevant_news,
        "present_timeline": present_timeline,
        "future_timelines": future_timelines,
    }
