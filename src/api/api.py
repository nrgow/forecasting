from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI


def load_events_stats_table() -> list[dict]:
    """Load the events stats table from the JSONL file."""
    data_path = Path(__file__).resolve().parents[2] / "data" / "events_stats_table.jsonl"
    lines = data_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the events stats table once at startup."""
    app.state.events_stats_table = load_events_stats_table()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/event_stats_table")
def event_stats_table() -> list[dict]:
    """Return the full events stats table."""
    return app.state.events_stats_table
