import logging
import os
import time

from openai import OpenAI

from ..mlflow_tracing import log_inference_call


def build_perplexity_timeline_prompt(
    topic_pertaining_to: str, current_date: str
) -> str:
    """Return the prompt used for Perplexity timeline generation."""
    return (
        "Construct a dense, grounded timeline,\n"
        "including concrete events relating to the topic,\n"
        "as well as statements made by the involved people and organizations,\n"
        "going back in time maximum past two years,\n"
        f"with heavy focus on the most recent events (today is {current_date}),\n"
        "for the following topic:\n"
        f'   "{topic_pertaining_to}".'
    )


def generate_present_timeline_perplexity(
    topic_pertaining_to: str,
    current_date: str,
    model: str = "perplexity/sonar-deep-research",
    temperature: float = 0.2,
    max_tokens: int = 12000,
    event_group_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Generate a present timeline via Perplexity on OpenRouter."""
    prompt = build_perplexity_timeline_prompt(topic_pertaining_to, current_date)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    logging.info("Perplexity timeline generation starting topic=%s", topic_pertaining_to)
    started_at = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - started_at
    timeline = completion.choices[0].message.content
    logging.info(
        "Perplexity timeline generation completed seconds=%.2f chars=%s",
        elapsed,
        len(timeline),
    )
    log_inference_call(
        name="present_timeline.perplexity",
        model=model,
        inputs={
            "topic_pertaining_to": topic_pertaining_to,
            "current_date": current_date,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        outputs={"timeline": timeline},
        metadata={
            "pipeline_run_id": run_id,
            "event_group_id": event_group_id,
            "stage": "perplexity",
        },
        duration_seconds=elapsed,
    )
    return {
        "prompt": prompt,
        "timeline": timeline,
        "model": model,
        "current_date": current_date,
    }
