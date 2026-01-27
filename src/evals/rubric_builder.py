import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..mlflow_tracing import log_inference_call


def build_rubric_from_article(
    article_text: str,
    topic: str,
    target_date: str,
    model: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.1,
    max_tokens: int = 600,
    base_url: str | None = None,
) -> str:
    """Generate a concrete scoring rubric from a gold-standard one-day article."""
    if base_url is None:
        base_url = "https://openrouter.ai/api/v1"
    client = OpenAI(
        base_url=base_url,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    prompt = (
        "You will derive a rubric to score one-day news summaries.\n"
        f"Topic: {topic}\n"
        f"Target date: {target_date}\n"
        "Gold-standard article (treat as 10/10 in every category):\n"
        f"{article_text}\n\n"
        "Write a concise rubric with 4–6 criteria. For each criterion include:\n"
        "- Name (2-4 words)\n"
        "- What good looks like (calibrated to the gold article)\n"
        "- Scoring bands: 10, 7, 4, 1 with short bullet guidance\n"
        "Keep total length under 220 words. Return plain markdown."
    )
    logging.info("Rubric generation starting model=%s", model)
    started_at = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - started_at
    content = completion.choices[0].message.content
    usage = completion.usage
    logging.info(
        "Rubric generation completed seconds=%.2f tokens=%s",
        elapsed,
        usage.total_tokens if usage is not None else None,
    )
    log_inference_call(
        name="one_day.rubric_builder",
        model=model,
        inputs={
            "topic": topic,
            "target_date": target_date,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        outputs={"rubric": content},
        metadata={
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage is not None else None,
                "completion_tokens": usage.completion_tokens if usage is not None else None,
                "total_tokens": usage.total_tokens if usage is not None else None,
            }
        },
        duration_seconds=elapsed,
    )
    return content


def load_article(path: Path) -> str:
    """Return article text from a file path."""
    return path.read_text(encoding="utf-8")
