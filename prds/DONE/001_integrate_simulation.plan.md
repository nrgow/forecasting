# Plan for 001_integrate_simulation

## Goals
- Add two simulation flows for `simulationActive` EventGroups:
  - PresentTimeline generation (once-off/occasional) and storage.
  - Near-real-time (every ~15 minutes) news scan, relevance judgment, FutureTimeline generation, event probability estimation, and storage.
- Persist news × eventgroup relevance judgments for future browsing/model training.
- Persist event probability estimates for future browsing/serving via the API.
- Keep storage file-based (JSONL) while abstracting storage for later DB migration.
- Avoid over- or under-abstracting; provide clean interfaces.

## Summary of existing context
- PresentTimeline generation logic exists: `src/simulation/generate_present_timeline.py`.
- FutureTimeline generation logic exists: `src/simulation/generate_future_timeline.py`.
- Active EventGroups list lives in `data/active_event_groups.jsonl` (dummy data with one entry).
- Pipeline entry point exists in `src/processing/pipeline.py`, with `run_pipeline` as a thin wrapper over file-path-based methods.

## Proposed architecture (high-level)
1. **EventGroupSource**: loads active EventGroups from JSONL.
2. **SimulationStorage**: interface for storing timelines, relevance judgments, probability estimates, and metadata (e.g., LLM args used for FutureTimelines).
3. **PresentTimelineService**: orchestrates timeline generation and storage.
4. **RealtimeEstimationService**: evaluates new articles against active EventGroups, stores judgments, and generates FutureTimelines.

## Storage design (filesystem, JSONL)
- `data/simulation/present_timelines.jsonl`
  - One record per EventGroup, includes `event_group_id`, `generated_at`, `timeline`, `derived_keyterms`, etc.
- `data/simulation/future_timelines.jsonl`
  - One record per EventGroup, includes `event_group_id`, `generated_at`, `timeline`.
- `data/simulation/estimated_event_probabilities.jsonl`
  - One record per EventGroup, includes `event_group_id`, `future_timelines`, `probability`.
- `data/simulation/realtime_relevance.jsonl`
  - One record per `(event_group_id, article_id)` judgment with timestamp and score/label.
- `data/simulation/simulation_runs.jsonl`
  - Tracks run metadata: run id, start/end time, counts, config, token cost.

Notes:
- Records should be append-only JSONL for simplicity.
- A small `SimulationStorage` class encapsulates read/write paths and schema, to ease later DB swap.

## Pipeline integration
- Keep `run_pipeline` minimal; logic lives in `src/processing/pipeline.py` methods that accept file paths.
- Add a simulation entry point method, e.g. `run_simulation_pipeline(...paths...)`, called from `run_pipeline`.
- Steps inside the simulation method:
  1. `load_active_event_groups(active_event_groups_path)`
  2. `generate_active_present_timelines(event_groups, storage_paths)`
  3. `generate_active_future_timelines(event_groups, new_articles_path, storage_paths)`

## PresentTimeline flow
- Load active EventGroups from JSONL.
- For each active EventGroup:
  - Check if a present timeline exists in storage to avoid rework.
  - If missing or stale, call `generate_present_timeline(event_group)` and store.
- Store derived keyterms if available from generation step.

## Near real-time flow
- Get new articles from news downloader output (via `download_news_past_week` in `src/processing/pipeline.py`).
- For each active EventGroup:
  - For each new article: compute relevance judgment (binary; LLM batch selection).
  - Store judgment record for future browsing/model training.
- Generate FutureTimeline(s) using the relevant articles and store.
- Estimate event probability from the FutureTimeline(s) and store.
- Deduplicate to avoid repeated judgments for the same `(event_group_id, article_id)`.

## Abstraction details
- `EventGroupSource`:
  - `list_active_event_groups()` returns structured objects.
- `SimulationStorage`:
  - `load_present_timeline(event_group_id)`
  - `save_present_timeline(record)`
  - `append_relevance_judgment(record)`
  - `save_future_timeline(record)`
  - `save_probability_estimate(record)`
  - `append_run_metadata(record)`
- `RealtimeEstimationService`:
  - `evaluate(event_groups, articles)` returns or writes judgments, FutureTimelines, and probabilities.

## Confirmed answers / constraints
- `event_group_id` is `EventGroup.id()` in `src/prediction_markets.py`.
- New articles are downloaded by `download_news_past_week` in `src/processing/pipeline.py`.
- Relevance judgment is binary using `src/simulation/news_relevance_dspy.py`.
- No ranking/embedding reuse for now.
- Retention policy: forever.
- PresentTimeline regeneration: manual trigger.

## Open questions
- How should “stale” be defined for PresentTimeline in manual-trigger mode?
- What is the minimal schema for probability estimates (e.g., `probability`, `explanation`, `model`)?
- Are FutureTimeline generations single-shot or should we store multiple candidates per run?
- Should deduplication be exact `(event_group_id, article_id)` or include timestamps to allow rejudgment?

## Implementation steps
1. Inspect active eventgroup schema and article storage format (confirm IDs, paths).
2. Add `SimulationStorage` and EventGroup source modules.
3. Implement PresentTimeline generation with idempotency and storage writes.
4. Implement near real-time relevance evaluation, FutureTimeline generation, and probability estimation.
5. Integrate into `src/processing/pipeline.py` via a new simulation method called by `run_pipeline`.
6. Add minimal logging and run metadata.

## Success criteria
- Running pipeline generates/stores PresentTimeline for each active EventGroup.
- Running pipeline stores relevance judgments, FutureTimelines, and probability estimates.
- Stored JSONL files can be read and browsed later by frontend.
- Abstractions are small and easy to migrate to DB later.
