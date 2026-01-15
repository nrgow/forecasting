import datetime as dt
import hashlib
import json
import logging
import re
import time
from pathlib import Path
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from ..prediction_markets import Events
from .storage import SimulationStorage


class OpenForecasterClient:
    """Hugging Face client for OpenForecaster generation."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        """Initialize the OpenForecaster pipeline with bf16 weights."""
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate(self, prompt: str) -> str:
        """Generate a raw completion for a single prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        stopping = StoppingCriteriaList(
            [
                ProbabilityTagStoppingCriteria(
                    tokenizer=self.tokenizer,
                    start_length=inputs["input_ids"].shape[1],
                )
            ]
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                stopping_criteria=stopping,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ProbabilityTagStoppingCriteria(StoppingCriteria):
    """Stop generation once a closing probability tag is produced."""

    def __init__(self, tokenizer: AutoTokenizer, start_length: int) -> None:
        """Track generated tokens after the prompt to find closing tags."""
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.generated_text = ""
        self.last_length = start_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: dict
    ) -> bool:
        """Return True when the closing probability tag appears."""
        current_length = input_ids.shape[1]
        if current_length <= self.last_length:
            return False
        new_ids = input_ids[0, self.last_length : current_length]
        self.generated_text += self.tokenizer.decode(new_ids, skip_special_tokens=True)
        self.last_length = current_length
        return "</probability>" in self.generated_text


def load_active_event_group_ids(path: Path) -> list[str]:
    """Load active event group ids from JSONL."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["event_group_id"] for line in lines]


def extract_probability(text: str) -> float | None:
    """Extract the probability from a <probability> tag."""
    match = re.search(r"<probability>(.*?)</probability>", text, flags=re.DOTALL)
    if match is None:
        return None
    return float(match.group(1).strip())


def extract_generated_only(prompt: str, generated_text: str) -> str:
    """Return the generated portion after the prompt."""
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].lstrip()
    return generated_text


def format_recent_news(records: list[dict], max_news: int) -> str:
    """Format relevant news records into a readable summary list."""
    records = sorted(
        records, key=lambda item: item["article_published_at"], reverse=True
    )
    items = []
    for idx, record in enumerate(records[:max_news], start=1):
        items.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"Published: {record['article_published_at']}",
                    f"Title: {record['article_title']}",
                    f"Summary: {record['article_desc']}",
                    f"URL: {record['article_url']}",
                ]
            )
        )
    return "\n\n".join(items)


def build_openforecaster_prompt(
    market: dict,
    present_timeline: str,
    relevant_news: list[dict],
    max_news: int,
) -> str:
    """Build an OpenForecaster prompt with timeline and news context."""
    retrieved_news_articles_summaries = format_recent_news(relevant_news, max_news)
    extra_info1 = ""
    extra_info2 = ""
    if len(retrieved_news_articles_summaries) > 10:
        extra_info1 = (
            " You will also be provided with a list of retrieved news articles summaries "
            "which you may refer to when coming up with your answer."
        )
        extra_info2 = (
            "\nRelevant passages from retrieved news articles:\n"
            f"{retrieved_news_articles_summaries}\n"
        )
    resolution_criteria = f"The market resolves by {market['end_date']}."
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened.{extra_info1} Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).
        
Question Title: {market['question']}
Question Background: {present_timeline.strip()}
Resolution Criteria: {resolution_criteria}
{extra_info2}
Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""
    return prompt


def run_openforecaster_pipeline(
    active_event_groups_path: Path,
    events_path: Path,
    storage_dir: Path,
    max_news: int = 8,
    max_new_tokens: int = 8192 * 4,
    temperature: float = 0.6,
    model_name: str = "nikhilchandak/OpenForecaster-8B",
) -> dict:
    """Generate OpenForecaster analyses for active event groups."""
    run_id = hashlib.sha256(
        dt.datetime.now(dt.timezone.utc).isoformat().encode("utf-8")
    ).hexdigest()
    logging.info("OpenForecaster run %s starting", run_id)
    storage = SimulationStorage(storage_dir)
    events = Events.from_file(events_path)
    event_group_index = {
        group.id(): group for group in events.group_by_title_template()
    }
    active_event_group_ids = load_active_event_group_ids(active_event_groups_path)
    present_timeline_index = storage.load_present_timeline_index()
    client = OpenForecasterClient(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    generated = 0
    for event_group_id in active_event_group_ids:
        event_group = event_group_index[event_group_id]
        logging.info(
            "OpenForecaster event group %s starting (%s)",
            event_group_id,
            event_group.template_title,
        )
        open_markets = [
            market
            for market in event_group.open_markets()
            if market.active and not market.closed
        ]
        present_record = present_timeline_index[event_group_id]
        present_timeline = present_record["timeline"]["merged"]["merged_timeline"]
        relevant_news = storage.load_latest_relevant_news(event_group_id)
        for market in open_markets:
            if market.end_date is None:
                raise ValueError(f"Open market {market.id} missing end_date.")
            prompt = build_openforecaster_prompt(
                market={
                    "id": market.id,
                    "question": market.question,
                    "end_date": market.end_date,
                },
                present_timeline=present_timeline,
                relevant_news=relevant_news,
                max_news=max_news,
            )
            generation_started = time.perf_counter()
            analysis = client.generate(prompt)
            generation_elapsed = time.perf_counter() - generation_started
            logging.info(
                "OpenForecaster market %s generated in %.2fs",
                market.id,
                generation_elapsed,
            )
            generated_only = extract_generated_only(prompt, analysis)
            probability = extract_probability(generated_only)
            record = {
                "event_group_id": event_group_id,
                "event_group_title": event_group.template_title,
                "market_id": market.id,
                "market_question": market.question,
                "market_slug": market.slug,
                "market_end_date": market.end_date,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "model": model_name,
                "prompt": prompt,
                "analysis": analysis,
                "generated_only": generated_only,
                "estimated_probability": probability,
                "valid": probability is not None,
                "run_id": run_id,
                "max_news": max_news,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "news_items": len(relevant_news),
            }
            storage.append_openforecaster_analysis(record)
            generated += 1
    logging.info("OpenForecaster run %s completed generated=%s", run_id, generated)
    return {"run_id": run_id, "generated": generated}
