import logging
import os
import time

from openai import OpenAI

from ..mlflow_tracing import log_inference_call


def build_recent_news_timeline_prompt(
    event_group_prompt: str, window_start: str, window_end: str
) -> str:
    """Return the prompt used for Perplexity recent news timelines."""
    return (
        "Summarize major news developments for the topic below between the given timestamps.\n"
        "Include both updates tied to the prediction market resolution criteria and\n"
        "broader, general news about the topic that is political/geopolitical in nature\n"
        "(e.g., statements, meetings, negotiations, legislation, policy actions, conflicts,\n"
        "or diplomatic moves) that could influence understanding of the topic.\n"
        "Output ONLY a concise timeline in chronological order with dated bullet points\n"
        "(YYYY-MM-DD - event). Do not include analysis, commentary, caveats, or\n"
        "references to search results or sources.\n"
        "If there were no major updates, output ONLY this single line:\n"
        f'"No major updates since {window_start}."\n\n'
        f"Time window: {window_start} to {window_end}\n\n"
        "Topic:\n"
        f"{event_group_prompt}"
    )


def generate_recent_news_timeline_perplexity(
    event_group_prompt: str,
    window_start: str,
    window_end: str,
    model: str = "perplexity/sonar",
    temperature: float = 0.2,
    max_tokens: int = 2500,
    event_group_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Generate a recent news timeline via Perplexity on OpenRouter."""
    prompt = build_recent_news_timeline_prompt(
        event_group_prompt=event_group_prompt,
        window_start=window_start,
        window_end=window_end,
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    logging.info(
        "Perplexity recent news generation starting window_start=%s window_end=%s",
        window_start,
        window_end,
    )
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
        "Perplexity recent news generation completed seconds=%.2f chars=%s",
        elapsed,
        len(timeline),
    )
    log_inference_call(
        name="recent_news.perplexity",
        model=model,
        inputs={
            "event_group_prompt": event_group_prompt,
            "window_start": window_start,
            "window_end": window_end,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        outputs={"timeline": timeline},
        metadata={
            "pipeline_run_id": run_id,
            "event_group_id": event_group_id,
            "stage": "recent_news",
        },
        duration_seconds=elapsed,
    )
    return {
        "prompt": prompt,
        "timeline": timeline,
        "model": model,
        "window_start": window_start,
        "window_end": window_end,
    }
