# Prediction Approach (Project Overview)

## Goal

Turn live news + structured prediction market data into near real-time probability estimates for open markets. The system groups related markets into EventGroups, builds a present-day narrative for context, filters incoming news for relevance, simulates future timelines, and aggregates implied answers into probabilities that can be served in the API/UI.

## Core Data Sources

- Prediction markets: Polymarket events and open markets via `src/prediction_markets.py` (cached + rate-limited API).
- News stream: GDELT GAL JSONL files downloaded to `/mnt/ssd/newstalk-data/gdelt-gal` via `src/news_data.py`.
- Background context: Wikipedia pages used in present timeline generation.

## Key Concepts

- EventGroup: A cluster of related Polymarket events grouped by a normalized title template (`EventGroup.id()` is a slug).
- Present timeline: A long-form narrative up to “now” for each EventGroup, used as context for future timeline generation.
- Relevance judgments: Article-level binary relevance labels per EventGroup.
- Future timelines: Simulated continuations of the present narrative.
- Market implications: For each simulated timeline, a yes/no answer to each open market question at that market’s resolution date.
- Probability estimates: Mean of implied answers across rollouts for each market.

## End-to-End Pipeline (current code path)

### 1) Fetch/Update Inputs

- `run_pipeline` in `src/processing/run_pipeline.py` downloads the latest GDELT files via `NewsDownloader.download_latest`.
- Polymarket fetching + event table steps exist in `src/processing/pipeline.py` but are currently commented out in `run_pipeline`.

### 2) Present Timeline Generation (optional/one-off)

- Entry: `run_present_timeline_pipeline` in `src/simulation/simulation_pipeline.py`.
- For each active EventGroup, `PresentTimelineService.generate_if_missing` calls `generate_present_timeline`:
  - Draft initial long-form timeline with `dspy.ReAct(EventsTimeline)` using Wikipedia tools.
  - Extract dated events with `dspy.Predict(ExtractEvents)`.
  - Expand each extracted event into a subtimeline (`SpecificEventsTimeline`).
  - Merge subtimelines chronologically (`MergeTimelines`).
- In `run_pipeline`, this step is currently commented out.

### 3) News Relevance Filtering (realtime gate)

- Entry: `run_relevance_pipeline` in `src/simulation/simulation_pipeline.py`.
- Articles are collected from GDELT files within the past 24 hours or since the last relevance run.
- Zero-shot pre-filtering (`ZeroShotClassifier`) is applied first; all scores are stored, but only items above a threshold continue to LLM relevance.
- `RelevanceJudgmentService.process` uses `NewsRelevanceJudge` (DSPy prompt `SelectRelevantArticles`) to mark articles as relevant or not, per EventGroup, storing every decision in `data/simulation/realtime_relevance.jsonl`.

### 4) Future Timeline Simulation + Probability Estimation

- Entry: `run_future_timeline_pipeline` in `src/simulation/simulation_pipeline.py`.
- For each EventGroup, relevant articles are converted to context strings (headline + description).
- The future timeline prompt is neutral and date-bounded:
  - Base instruction: “Continue the timeline from the current date using the provided contexts.”
  - End date: latest open market end date for the group.
- `FutureTimelineEstimator.generate` calls `run_model` in `src/simulation/generate_future_timeline.py`:
  - `FutureTimeline` generates a simulated timeline given scenario + contexts.
  - `TimelineImplication` answers each open market question at its own `implication_date`.
  - Rollouts are run across models + temperatures; each rollout stores model, temp, and implied answers.
- `aggregate_market_probabilities` averages implied answers into per-market probabilities and stores them in `data/simulation/estimated_event_probabilities.jsonl`.

## Storage Layout (JSONL)

- `data/simulation/present_timelines.jsonl`: Present timeline artifacts per EventGroup.
- `data/simulation/realtime_relevance.jsonl`: Article relevance judgments.
- `data/simulation/news_zero_shot.jsonl`: Zero-shot scores and thresholds.
- `data/simulation/future_timelines.jsonl`: Full future timeline rollouts + market implications.
- `data/simulation/estimated_event_probabilities.jsonl`: Aggregated probabilities per market.
- `data/simulation/simulation_runs.jsonl`: Run metadata.

## Serving / UI

- `src/api/api.py` loads stored JSONL at startup and serves:
  - `/event_stats_table`: events table with active flags.
  - `/event_groups/{event_group_id}`: present timeline, relevant news, future timelines, and market probabilities.
  - `/open_market_opportunities`: markets ranked by delta between Polymarket price and model estimate.

## Current Approach Summary

1. Identify active EventGroups (from `data/active_event_groups.jsonl`).
2. Optionally build long-form present timelines (Wikipedia-backed).
3. Continuously ingest news; pre-filter with zero-shot classifier.
4. Use LLM relevance to select article context per EventGroup.
5. Simulate multiple future timelines; answer each market question on its resolution date.
6. Aggregate implied answers into probabilities and serve via API/UI.
