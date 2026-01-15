from functools import partial
from multiprocessing import Pool
import logging
import time

import dspy
import wikipedia

from ..mlflow_tracing import configure_dspy_autolog, log_inference_call


class FutureTimeline(dspy.Signature):
    """Continue a realistic chronological timeline from the current date using the provided contexts."""

    timeline_scenario: str = dspy.InputField()
    contexts: list[str] = dspy.InputField()
    current_date: str = dspy.InputField()
    simulated_timeline: str = dspy.OutputField()


class TimelineImplication(dspy.Signature):
    """Given a simulated future timeline and a question, return whether the timeline implies the answer is true as of the specified date."""

    timeline: str = dspy.InputField()
    question_to_answer: str = dspy.InputField()
    implication_date: str = dspy.InputField()
    implied_answer: bool = dspy.OutputField(
        desc="The answer to the question, given the timeline"
    )


class SanitizeFutureTimelineTopic(dspy.Signature):
    """Summarize event details into a one-sentence general topic without market framing."""

    event_title: str = dspy.InputField()
    event_descriptions: list[str] = dspy.InputField()
    sanitized_topic: str = dspy.OutputField(
        desc="One sentence topic, without market framing or resolution criteria"
    )


def sanitize_future_timeline_topic(
    event_title: str,
    event_descriptions: list[str],
    model: str,
    trace_metadata: dict | None = None,
) -> str:
    """Generate a sanitized one-sentence topic for future timeline prompts."""
    configure_dspy_autolog()
    dspy.configure(lm=dspy.LM(model))
    predictor = dspy.Predict(SanitizeFutureTimelineTopic)
    started_at = time.perf_counter()
    result = predictor(
        event_title=event_title,
        event_descriptions=event_descriptions,
    )
    elapsed = time.perf_counter() - started_at
    logging.info("Future timeline topic sanitization completed seconds=%.2f", elapsed)
    log_inference_call(
        name="future_timeline.sanitize_topic",
        model=model,
        inputs={
            "event_title": event_title,
            "event_descriptions": event_descriptions,
        },
        outputs={"sanitized_topic": result.sanitized_topic.strip()},
        metadata={"stage": "sanitize_topic", **(trace_metadata or {})},
        duration_seconds=elapsed,
    )
    return result.sanitized_topic.strip()


def run_model(
    model,
    scenario,
    contexts,
    current_date,
    market_specs,
    temps=[0.1, 0.3, 0.5, 0.7, 0.9],
    rollouts_per_temp: int = 1,
    trace_metadata: dict | None = None,
):
    configure_dspy_autolog()
    dspy.configure(lm=dspy.LM(model))
    rval = []
    rollout_id = 0
    for temp in temps:
        for _ in range(rollouts_per_temp):
            timeline_prediction = dspy.Predict(
                FutureTimeline, rollout_id=rollout_id, temperature=temp
            )
            question_answering = dspy.ChainOfThought(TimelineImplication, temperature=0)

            timeline_started_at = time.perf_counter()
            timeline = timeline_prediction(
                timeline_scenario=scenario,
                contexts=contexts,
                current_date=current_date,
            )
            timeline_elapsed = time.perf_counter() - timeline_started_at
            log_inference_call(
                name="future_timeline.generate",
                model=model,
                inputs={
                    "timeline_scenario": scenario,
                    "contexts": contexts,
                    "current_date": current_date,
                    "temperature": temp,
                    "rollout_id": rollout_id,
                },
                outputs={"simulated_timeline": timeline.simulated_timeline},
                metadata={
                    **(trace_metadata or {}),
                    "stage": "future_timeline",
                    "rollout_id": rollout_id,
                    "temperature": temp,
                },
                duration_seconds=timeline_elapsed,
            )
            market_implications = []
            for market in market_specs:
                implication_started_at = time.perf_counter()
                timeline_implication = question_answering(
                    timeline=timeline.simulated_timeline,
                    question_to_answer=market["market_question"],
                    implication_date=market["implication_date"],
                )
                implication_elapsed = time.perf_counter() - implication_started_at
                log_inference_call(
                    name="future_timeline.implication",
                    model=model,
                    inputs={
                        "timeline": timeline.simulated_timeline,
                        "question_to_answer": market["market_question"],
                        "implication_date": market["implication_date"],
                    },
                    outputs={"implied_answer": timeline_implication.implied_answer},
                    metadata={
                        **(trace_metadata or {}),
                        "stage": "timeline_implication",
                        "market_id": market["market_id"],
                        "market_slug": market["market_slug"],
                        "rollout_id": rollout_id,
                    },
                    duration_seconds=implication_elapsed,
                )
                market_implications.append(
                    {
                        "market_id": market["market_id"],
                        "market_question": market["market_question"],
                        "market_slug": market["market_slug"],
                        "market_end_date": market["market_end_date"],
                        "implication_date": market["implication_date"],
                        "implied_answer": timeline_implication.implied_answer,
                    }
                )
            rval.append(
                {
                    "model": model,
                    "rollout_id": rollout_id,
                    "temp": temp,
                    "simulated_timeline": timeline.simulated_timeline,
                    "market_implications": market_implications,
                }
            )
            rollout_id += 1

    return rval


def generate_future_timeline(
    scenario: str,
    wiki_context_pages: list[str],
    current_date: str,
    market_specs: list[dict],
    rollouts_per_temp: int = 1,
):
    contexts = []
    for page_title in wiki_context_pages:
        page = wikipedia.page(page_title)
        context = page.title + "\n" + page.content
        contexts.append(context)

    models = [
        # "openrouter/anthropic/claude-opus-4.5",
        # "openrouter/openrouter/bert-nebulon-alpha",
        "openrouter/x-ai/grok-4.1-fast",
        # "openrouter/google/gemini-3-pro-preview",
        # "openrouter/openai/gpt-5.1",
    ]
    prediction_function = partial(
        run_model,
        scenario=scenario,
        contexts=contexts,
        current_date=current_date,
        market_specs=market_specs,
        rollouts_per_temp=rollouts_per_temp,
    )

    results = []
    with Pool(processes=len(models)) as pool:
        processed = pool.map(prediction_function, models)
        for model_results in processed:
            for result in model_results:
                results.append(result)
    return results
