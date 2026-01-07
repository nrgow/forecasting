# Newstalk

## API

1. Install Python dependencies:

```bash
uv sync
```

2. Start the API:

```bash
uv run uvicorn src.api.api:app --reload
```

The API loads `events_stats_table.jsonl` on startup and serves it at
`http://localhost:8000/event_stats_table`.

## UI

1. Install frontend dependencies:

```bash
cd frontend
npm install
```

2. Start the UI:

```bash
npm run dev
```

Vite proxies `/event_stats_table` to the FastAPI server at
`http://localhost:8000`.
