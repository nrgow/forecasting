# Prediction Approach (Present + Future Timelines)

## Present Timeline (context builder)

- Entry point: `PresentTimelineService.generate_if_missing` in `src/simulation/simulation_pipeline.py`.
- For each active `EventGroup`, the service generates a present-day timeline if none exists (or if forced), then stores it via `SimulationStorage.append_present_timeline`.
- Timeline generation in `src/simulation/generate_present_timeline.py`:
  - Drafts an initial long-form timeline with `dspy.ReAct(EventsTimeline)` using cached Wikipedia tools.
  - Extracts a bounded set of dated events via `dspy.Predict(ExtractEvents)`.
  - Expands each extracted event into its own subtimeline using `dspy.ReAct(SpecificEventsTimeline)`.
  - Merges subtimelines into a single chronological narrative with `dspy.ChainOfThought(MergeTimelines)`.
- The extracted event descriptions are used as `derived_keyterms` and stored alongside the full timeline output.

## Realtime Relevance Filtering (article gating)

- Entry point: `RelevanceJudgmentService.process` in `src/simulation/simulation_pipeline.py`.
- For each `EventGroup`, incoming `NewsArticle` items are batched and judged for relevance using `NewsRelevanceJudge` (LLM prompt via `SelectRelevantArticles`).
- Relevance decisions are persisted with per-article metadata (IDs, timestamps, model, run_id).
- Only relevant articles are passed downstream; their headline + description strings become the context set for future timelines.

## Future Timeline Simulation (probability estimation)

- Prompt assembly: `build_future_timeline_prompt` in `src/simulation/simulation_pipeline.py` frames the future timeline question and bounds it by the market end date.
- Entry point: `FutureTimelineService.process` in `src/simulation/simulation_pipeline.py`.
- `FutureTimelineEstimator.generate` runs multiple models and temperatures, delegating to `run_model` in `src/simulation/generate_future_timeline.py`.
- For each model/temperature rollout:
  - `FutureTimeline` generates a simulated future timeline from the scenario, relevant article contexts, and current date.
  - `TimelineImplication` answers whether the simulated timeline implies the market question is true.
  - The rollout result records the timeline text and a boolean `choice`.
- The final probability estimate is the mean of boolean choices across rollouts: `sum(int(choice)) / len(choices)`.
- Both raw rollout results and the aggregated probability are stored via `SimulationStorage.append_future_timeline` and `append_probability_estimate`.

## End-to-End Flow Summary

1. Generate or load present timelines to ground the event context.
2. Judge incoming news relevance per event group; store decisions.
3. For each event group with relevant articles from the relevance run:
   - Build a bounded future-timeline prompt.
   - Run multi-rollout simulations across model/temperature settings.
   - Convert the set of implied-true/false answers into a probability.
4. Persist timelines and probability estimates for downstream API serving.
