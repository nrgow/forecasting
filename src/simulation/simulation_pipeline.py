import datetime as dt
import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Iterable

import dspy
import more_itertools as mit

from ..prediction_markets import EventGroup, Events, parse_open_market_end_date
from ..news_enrichment import ZeroShotClassifier
from .generate_future_timeline import run_model, sanitize_future_timeline_topic
from .generate_present_timeline import generate_present_timeline
from .news_relevance_dspy import SelectRelevantArticles
from .storage import SimulationStorage


@dataclass(frozen=True)
class NewsArticle:
    """Normalized news article representation for simulation."""

    article_id: str
    url: str
    title: str
    desc: str
    published_at: dt.datetime
    domain: str
    outlet_name: str
    lang: str

    def prompt_text(self) -> str:
        """Return the text used for relevance judgment."""
        return f"{self.title}\n\n{self.desc}".strip()


class NewsRelevanceJudge:
    """LLM-based relevance selector for news articles."""

    def __init__(self, model: str) -> None:
        """Configure the LLM model for relevance selection."""
        self.model = model
        dspy.configure(lm=dspy.LM(model))
        self.module = dspy.Predict(SelectRelevantArticles)

    def select_relevant(self, query: str, articles: list[str]) -> list[str]:
        """Return the subset of article strings judged relevant to the query."""
        dspy.configure(lm=dspy.LM(self.model))
        result = self.module(query_or_article=query, articles=articles)
        return result.relevant_articles


class FutureTimelineEstimator:
    """Generate future timelines and implied probability estimates."""

    def __init__(
        self,
        models: list[str],
        temps: list[float],
        rollouts_per_temp: int,
    ) -> None:
        """Initialize the estimator with model, temperature, and rollout settings."""
        self.models = models
        self.temps = temps
        self.rollouts_per_temp = rollouts_per_temp

    def generate(
        self,
        scenario: str,
        contexts: list[str],
        current_date: str,
        market_specs: list[dict],
    ) -> dict:
        """Generate future timeline rollouts and market-level probabilities."""
        results = list(
            chain.from_iterable(
                run_model(
                    model=model,
                    scenario=scenario,
                    contexts=contexts,
                    current_date=current_date,
                    market_specs=market_specs,
                    temps=self.temps,
                    rollouts_per_temp=self.rollouts_per_temp,
                )
                for model in self.models
            )
        )
        market_probabilities = aggregate_market_probabilities(results, market_specs)
        return {"results": results, "market_probabilities": market_probabilities}


def recurse_types(js):
    if isinstance(js, list):
        return [recurse_types(el) for el in js]
    if isinstance(js, dict):
        return {k: recurse_types(v) for k,v in js.items()}
    else:
        return type(js)

class PresentTimelineService:
    """Generate and store present timelines for active event groups."""

    def __init__(self, storage: SimulationStorage) -> None:
        """Initialize the service with storage."""
        self.storage = storage

    def generate_if_missing(
        self,
        event_groups: list[EventGroup],
        force: bool,
        target_timeline_chars: int | None,
        min_events: int | None,
        max_events: int | None,
    ) -> list[dict]:
        """Generate present timelines if missing or forced."""
        existing = self.storage.load_present_timeline_index()
        generated = []
        for event_group in event_groups:
            logging.info("Present timeline check for event_group_id=%s", event_group.id())
            if event_group.id() in existing and not force:
                logging.info(
                    "Present timeline exists for event_group_id=%s; skipping",
                    event_group.id(),
                )
                continue
            logging.info(
                "Generating present timeline for event_group_id=%s",
                event_group.id(),
            )
            output = generate_present_timeline(
                event_group.template_title,
                target_timeline_chars=target_timeline_chars
                if target_timeline_chars is not None
                else 24000,
                min_events=min_events if min_events is not None else 18,
                max_events=max_events if max_events is not None else 28,
            )
            extracted_events = output["extracted_events"]["events"]
            derived_keyterms = [event["description"] for event in extracted_events]
            record = {
                "event_group_id": event_group.id(),
                "event_group_title": event_group.template_title,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "timeline": output,
                "derived_keyterms": derived_keyterms,
            }
            logging.info(f"{recurse_types(output)=}")            
            self.storage.append_present_timeline(record)
            logging.info(
                "Stored present timeline for event_group_id=%s",
                event_group.id(),
            )
            generated.append(record)
        return generated


class RelevanceJudgmentService:
    """Evaluate new articles for relevance against event groups."""

    def __init__(
        self,
        storage: SimulationStorage,
        relevance_model: str,
        batch_size: int,
    ) -> None:
        """Initialize the service with relevance model and batch sizing."""
        self.storage = storage
        self.judge = NewsRelevanceJudge(relevance_model)
        self.batch_size = batch_size

    def process(
        self,
        event_groups: list[EventGroup],
        articles: list[NewsArticle],
        run_id: str,
    ) -> list[dict]:
        """Judge relevance and store results."""
        logging.info(
            "Relevance judgment starting event_groups=%s articles=%s batch_size=%s",
            len(event_groups),
            len(articles),
            self.batch_size,
        )
        existing = self.storage.load_relevance_index()
        summaries = []
        for event_group in event_groups:
            event_group_id = event_group.id()
            candidates = [
                article
                for article in articles
                if (event_group_id, article.article_id) not in existing
            ]
            if not candidates:
                logging.info(
                    "Relevance skipping event_group_id=%s; no candidates",
                    event_group_id,
                )
                continue
            total_batches = (len(candidates) + self.batch_size - 1) // self.batch_size
            logging.info(
                "Relevance judging event_group_id=%s candidates=%s batches=%s",
                event_group_id,
                len(candidates),
                total_batches,
            )
            relevant_articles = 0
            query = build_event_group_prompt(event_group)
            for batch_index, batch in enumerate(
                mit.chunked(candidates, self.batch_size), start=1
            ):
                batch_list = list(batch)
                batch_texts = [article.prompt_text() for article in batch_list]
                started_at = time.perf_counter()
                relevant_texts = set(self.judge.select_relevant(query, batch_texts))
                elapsed = time.perf_counter() - started_at
                items_per_sec = len(batch_list) / elapsed
                relevant_in_batch = 0
                for article in batch_list:
                    relevant = article.prompt_text() in relevant_texts
                    record = {
                        "event_group_id": event_group_id,
                        "article_id": article.article_id,
                        "article_url": article.url,
                        "article_title": article.title,
                        "article_desc": article.desc,
                        "article_published_at": article.published_at.isoformat(),
                        "relevant": relevant,
                        "judged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "model": self.judge.model,
                        "run_id": run_id,
                    }
                    self.storage.append_relevance_judgment(record)
                    if relevant:
                        relevant_in_batch += 1
                        relevant_articles += 1
                logging.info(
                    "Relevance batch event_group_id=%s batch=%s/%s batch_size=%s relevant=%s",
                    event_group_id,
                    batch_index,
                    total_batches,
                    len(batch_list),
                    relevant_in_batch,
                )
                logging.info(
                    "Relevance throughput event_group_id=%s batch=%s/%s items=%s seconds=%.2f items_per_sec=%.2f",
                    event_group_id,
                    batch_index,
                    total_batches,
                    len(batch_list),
                    elapsed,
                    items_per_sec,
                )
            summaries.append(
                {
                    "event_group_id": event_group_id,
                    "articles_considered": len(candidates),
                    "relevant_articles": relevant_articles,
                }
            )
        return summaries


class FutureTimelineService:
    """Generate future timelines from relevance judgments."""

    def __init__(
        self,
        storage: SimulationStorage,
        timeline_models: list[str],
        timeline_temps: list[float],
        timeline_rollouts: int,
    ) -> None:
        """Initialize the service with timeline models, temperatures, and rollouts."""
        self.storage = storage
        self.estimator = FutureTimelineEstimator(
            timeline_models,
            timeline_temps,
            timeline_rollouts,
        )

    def process(
        self,
        event_groups: list[EventGroup],
        relevance_run_id: str,
        run_id: str,
        force_without_relevance: bool = False,
    ) -> list[dict]:
        """Generate future timelines for event groups with relevant articles."""
        relevance_records = self.storage.load_relevance_records(relevance_run_id)
        grouped_records: dict[str, list[dict]] = {}
        for record in relevance_records:
            grouped_records.setdefault(record["event_group_id"], []).append(record)

        summaries = []
        for event_group in event_groups:
            event_group_id = event_group.id()
            if event_group_id not in grouped_records:
                if not force_without_relevance:
                    logging.info(
                        "Future timeline skipping event_group_id=%s; no relevance records",
                        event_group_id,
                    )
                    continue
                records = self.storage.load_latest_relevance_records_for_group(event_group_id)
                if records:
                    logging.info(
                        "Future timeline using prior relevance records event_group_id=%s records=%s",
                        event_group_id,
                        len(records),
                    )
                else:
                    logging.info(
                        "Future timeline forcing event_group_id=%s; no relevance records",
                        event_group_id,
                    )
            else:
                records = grouped_records[event_group_id]
            relevant_records = [record for record in records if record["relevant"]]
            if not relevant_records and not force_without_relevance:
                logging.info(
                    "Future timeline skipping event_group_id=%s; no relevant records",
                    event_group_id,
                )
                continue
            if not relevant_records and force_without_relevance:
                logging.info(
                    "Future timeline forcing event_group_id=%s; no relevant records",
                    event_group_id,
                )
            logging.info(
                "Future timeline generating event_group_id=%s relevant_records=%s",
                event_group_id,
                len(relevant_records),
            )
            scenario = build_future_timeline_prompt(event_group, self.estimator.models[0])
            contexts = list(
                islice((prompt_text_from_record(record) for record in relevant_records), 50)
            )
            market_specs = build_market_implication_specs(event_group)
            if not market_specs:
                logging.info(
                    "Future timeline skipping event_group_id=%s; no open markets",
                    event_group_id,
                )
                continue
            timeline_output = self.estimator.generate(
                scenario=scenario,
                contexts=contexts,
                current_date=dt.date.today().isoformat(),
                market_specs=market_specs,
            )
            generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
            timeline_record = {
                "event_group_id": event_group_id,
                "event_group_title": event_group.template_title,
                "generated_at": generated_at,
                "scenario": scenario,
                "contexts_count": len(contexts),
                "models": self.estimator.models,
                "temps": self.estimator.temps,
                "rollouts_per_temp": self.estimator.rollouts_per_temp,
                "results": timeline_output["results"],
                "market_probabilities": timeline_output["market_probabilities"],
                "relevance_run_id": relevance_run_id,
                "run_id": run_id,
            }
            self.storage.append_future_timeline(timeline_record)
            for market_probability in timeline_output["market_probabilities"]:
                probability_record = {
                    "event_group_id": event_group_id,
                    "event_group_title": event_group.template_title,
                    "generated_at": generated_at,
                    "market_id": market_probability["market_id"],
                    "market_question": market_probability["market_question"],
                    "market_slug": market_probability["market_slug"],
                    "market_end_date": market_probability["market_end_date"],
                    "implication_date": market_probability["implication_date"],
                    "probability": market_probability["probability"],
                    "samples": market_probability["samples"],
                    "relevance_run_id": relevance_run_id,
                    "run_id": run_id,
                }
                self.storage.append_probability_estimate(probability_record)
            summaries.append(
                {
                    "event_group_id": event_group_id,
                    "relevant_articles": len(relevant_records),
                    "contexts_used": len(contexts),
                }
            )
        return summaries


def load_active_event_group_ids(path: Path) -> list[str]:
    """Load active event group ids from JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line)["event_group_id"] for line in handle]


def load_event_group_index(events_path: Path) -> dict[str, EventGroup]:
    """Load EventGroups and return an index by event_group_id."""
    events = Events.from_file(events_path)
    return {group.id(): group for group in events.group_by_title_template()}


def build_event_group_prompt(event_group: EventGroup) -> str:
    """Build a prompt string describing an event group."""
    descriptions = [event.description for event in event_group.events if event.description]
    unique_descriptions = list(mit.unique_everseen(descriptions))
    description_text = "\n\n".join(unique_descriptions)
    return f"{event_group.template_title}\n\n{description_text}".strip()


def prompt_text_from_record(record: dict) -> str:
    """Return prompt text for a relevance record."""
    return f"{record['article_title']}\n\n{record['article_desc']}".strip()


def build_sanitized_future_topic(event_group: EventGroup, model: str) -> str:
    """Return a sanitized topic sentence for future timeline prompts."""
    descriptions = [event.description for event in event_group.events if event.description]
    unique_descriptions = list(mit.unique_everseen(descriptions))
    return sanitize_future_timeline_topic(
        event_group.template_title, unique_descriptions, model
    )


def build_future_timeline_prompt(event_group: EventGroup, model: str) -> str:
    """Build a neutral prompt for future timeline generation bounded by market end dates."""
    base_prompt = (
        "Continue the timeline from the current date using the provided contexts."
    )
    latest_end_date = event_group.latest_open_market_end_date()
    if latest_end_date is None:
        return base_prompt
    event_date = latest_end_date.date().isoformat()
    return f"{base_prompt}\n\nGenerate the future timeline only up to {event_date}."


def build_market_implication_specs(event_group: EventGroup) -> list[dict]:
    """Build market implication specs with resolution dates for open markets."""
    specs = []
    for market in event_group.open_markets():
        if not market.active or market.closed:
            continue
        if market.end_date is None:
            raise ValueError(f"Open market {market.id} missing end_date.")
        end_date = parse_open_market_end_date(market.end_date).date().isoformat()
        specs.append(
            {
                "market_id": market.id,
                "market_question": market.question,
                "market_slug": market.slug,
                "market_end_date": market.end_date,
                "implication_date": end_date,
            }
        )
    return specs


def aggregate_market_probabilities(
    results: list[dict], market_specs: list[dict]
) -> list[dict]:
    """Aggregate implied answers into market probability estimates."""
    probabilities = []
    for market in market_specs:
        choices = []
        for result in results:
            for implication in result["market_implications"]:
                if implication["market_id"] == market["market_id"]:
                    choices.append(implication["implied_answer"])
                    break
        probability = sum(int(choice) for choice in choices) / len(choices) if choices else None
        probabilities.append(
            {
                "market_id": market["market_id"],
                "market_question": market["market_question"],
                "market_slug": market["market_slug"],
                "market_end_date": market["market_end_date"],
                "implication_date": market["implication_date"],
                "probability": probability,
                "samples": len(choices),
            }
        )
    return probabilities


def parse_news_datetime(value: str) -> dt.datetime:
    """Parse GDELT datetime values into timezone-aware datetimes."""
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_news_filename_datetime(path: Path) -> dt.datetime:
    """Parse the datetime from a GDELT filename."""
    ts = path.name.split(".")[0]
    return dt.datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)


def iter_news_paths(news_base_path: Path, since: dt.datetime | None) -> list[Path]:
    """Return news file paths filtered by filename timestamp."""
    paths = sorted(news_base_path.glob("*.gal.json.gz"))
    if since is None:
        return paths
    return [path for path in paths if parse_news_filename_datetime(path) >= since]


def make_article_id(url: str) -> str:
    """Return a stable article id from the URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def iter_news_articles(
    news_paths: Iterable[Path], since: dt.datetime | None
) -> Iterable[NewsArticle]:
    """Yield normalized English articles from GDELT files."""
    for path in news_paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["lang"] != "en":
                    continue
                published_at = parse_news_datetime(record["date"])
                if since is not None and published_at < since:
                    continue
                yield NewsArticle(
                    article_id=make_article_id(record["url"]),
                    url=record["url"],
                    title=record["title"],
                    desc=record["desc"],
                    published_at=published_at,
                    domain=record["domain"],
                    outlet_name=record["outletName"],
                    lang=record["lang"],
                )


def apply_zero_shot_filter(
    articles: list[NewsArticle],
    storage: SimulationStorage,
    run_id: str,
    class_name: str,
    min_probability: float,
) -> list[NewsArticle]:
    """Score articles with zero-shot classifier, persist results, and filter by score."""
    if not articles:
        return []
    classifier = ZeroShotClassifier(class_name=class_name)
    classifier.setup()
    try:
        indices: list[int] = []
        texts: list[str] = []
        for idx, article in enumerate(articles):
            text = article.prompt_text()
            if text.strip():
                indices.append(idx)
                texts.append(text)
        started_at = time.perf_counter()
        scores = classifier.predict_many(texts) if texts else []
        elapsed = time.perf_counter() - started_at
        if texts:
            items_per_sec = len(texts) / elapsed
            logging.info(
                "Zero-shot throughput class_name=%s items=%s seconds=%.2f items_per_sec=%.2f",
                class_name,
                len(texts),
                elapsed,
                items_per_sec,
            )
    finally:
        classifier.shutdown()
    score_map = {idx: score for idx, score in zip(indices, scores)}
    filtered: list[NewsArticle] = []
    for idx, article in enumerate(articles):
        score = score_map.get(idx, 0)
        record = {
            "article_id": article.article_id,
            "article_url": article.url,
            "article_title": article.title,
            "article_desc": article.desc,
            "article_published_at": article.published_at.isoformat(),
            "article_domain": article.domain,
            "article_outlet_name": article.outlet_name,
            "article_lang": article.lang,
            "zero_shot_class_name": class_name,
            "zero_shot_class_key": classifier.class_name_key,
            "zero_shot_score": score,
            "zero_shot_threshold": min_probability,
            "zero_shot_passed": score > min_probability,
            "run_id": run_id,
            "scored_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        storage.append_zero_shot_record(record)
        if score > min_probability:
            filtered.append(article)
    return filtered


def run_present_timeline_pipeline(
    active_event_groups_path: Path,
    events_path: Path,
    storage_dir: Path,
    force_present: bool = False,
    target_timeline_chars: int | None = None,
    min_events: int | None = None,
    max_events: int | None = None,
) -> dict:
    """Run the present timeline generation pipeline."""
    storage = SimulationStorage(storage_dir)
    run_id = hashlib.sha256(dt.datetime.now(dt.timezone.utc).isoformat().encode("utf-8")).hexdigest()
    run_started_at = dt.datetime.now(dt.timezone.utc)
    logging.info("Present timeline run %s starting", run_id)

    active_event_group_ids = load_active_event_group_ids(active_event_groups_path)
    event_group_index = load_event_group_index(events_path)
    event_groups = [event_group_index[event_group_id] for event_group_id in active_event_group_ids]
    logging.info("Present timeline event_groups=%s", len(event_groups))

    present_service = PresentTimelineService(storage)
    generated_present = present_service.generate_if_missing(
        event_groups,
        force_present,
        target_timeline_chars,
        min_events,
        max_events,
    )

    run_finished_at = dt.datetime.now(dt.timezone.utc)
    run_record = {
        "run_id": run_id,
        "started_at": run_started_at.isoformat(),
        "ended_at": run_finished_at.isoformat(),
        "present_timelines_generated": len(generated_present),
    }
    storage.append_run_metadata(run_record)
    logging.info("Present timeline run %s completed", run_id)
    return run_record


def run_realtime_simulation_pipeline(
    active_event_groups_path: Path,
    events_path: Path,
    news_base_path: Path,
    storage_dir: Path,
    relevance_model: str = "openrouter/x-ai/grok-4.1-fast",
    timeline_models: list[str] | None = ["openrouter/x-ai/grok-4.1-fast"],
    timeline_temps: list[float] | None = [0.7],
    batch_size: int = 25,
) -> dict:
    """Run the real-time simulation pipeline for active event groups."""
    logging.info("Real-time simulation pipeline starting")
    relevance = run_relevance_pipeline(
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        news_base_path=news_base_path,
        storage_dir=storage_dir,
        relevance_model=relevance_model,
        batch_size=batch_size,
    )
    future = run_future_timeline_pipeline(
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        storage_dir=storage_dir,
        relevance_run_id=relevance["run_id"],
        timeline_models=timeline_models,
        timeline_temps=timeline_temps,
    )
    return {"relevance": relevance, "future": future}


def run_relevance_pipeline(
    active_event_groups_path: Path,
    events_path: Path,
    news_base_path: Path,
    storage_dir: Path,
    relevance_model: str = "openrouter/x-ai/grok-4.1-fast",
    zero_shot_min_probability: float = 0.5,
    zero_shot_class_name: str = "international politics geopolitics world financial markets",
    batch_size: int = 25,
) -> dict:
    """Run the relevance judgment pipeline for active event groups."""
    storage = SimulationStorage(storage_dir)
    run_id = hashlib.sha256(dt.datetime.now(dt.timezone.utc).isoformat().encode("utf-8")).hexdigest()
    run_started_at = dt.datetime.now(dt.timezone.utc)
    logging.info("Relevance run %s starting", run_id)
    last_judged_at = storage.last_relevance_judged_at()
    cutoff = run_started_at - dt.timedelta(hours=24)
    if last_judged_at is None:
        news_since = cutoff
    else:
        last_judged_time = dt.datetime.fromisoformat(last_judged_at)
        news_since = max(last_judged_time, cutoff)
    logging.info("Relevance news_since=%s", news_since.isoformat())

    active_event_group_ids = load_active_event_group_ids(active_event_groups_path)
    event_group_index = load_event_group_index(events_path)
    event_groups = [event_group_index[event_group_id] for event_group_id in active_event_group_ids]
    logging.info("Relevance event_groups=%s", len(event_groups))

    news_paths = iter_news_paths(news_base_path, news_since)
    articles = list(iter_news_articles(news_paths, news_since))
    articles_total = len(articles)
    articles = apply_zero_shot_filter(
        articles=articles,
        storage=storage,
        run_id=run_id,
        class_name=zero_shot_class_name,
        min_probability=zero_shot_min_probability,
    )
    logging.info(
        "Relevance articles=%s news_paths=%s batch_size=%s",
        len(articles),
        len(news_paths),
        batch_size,
    )

    relevance_service = RelevanceJudgmentService(
        storage=storage,
        relevance_model=relevance_model,
        batch_size=batch_size,
    )
    summaries = relevance_service.process(event_groups, articles, run_id)

    run_finished_at = dt.datetime.now(dt.timezone.utc)
    run_record = {
        "run_id": run_id,
        "started_at": run_started_at.isoformat(),
        "ended_at": run_finished_at.isoformat(),
        "news_since": news_since.isoformat(),
        "news_until": run_finished_at.isoformat(),
        "articles_total": articles_total,
        "articles_considered": len(articles),
        "event_group_summaries": summaries,
        "relevance_model": relevance_model,
        "zero_shot_class_name": zero_shot_class_name,
        "zero_shot_min_probability": zero_shot_min_probability,
    }
    storage.append_run_metadata(run_record)
    logging.info("Relevance run %s completed", run_id)
    return run_record


def run_future_timeline_pipeline(
    active_event_groups_path: Path,
    events_path: Path,
    storage_dir: Path,
    relevance_run_id: str | None = None,
    timeline_models: list[str] | None = ["openrouter/x-ai/grok-4.1-fast"],
    timeline_temps: list[float] | None = [0.7],
    timeline_rollouts: int | None = 10,
    force_without_relevance: bool = False,
) -> dict:
    """Run the future timeline pipeline for active event groups."""
    storage = SimulationStorage(storage_dir)
    run_id = hashlib.sha256(dt.datetime.now(dt.timezone.utc).isoformat().encode("utf-8")).hexdigest()
    run_started_at = dt.datetime.now(dt.timezone.utc)
    logging.info("Future timeline run %s starting", run_id)
    if relevance_run_id is None:
        relevance_run_id = storage.last_relevance_run_id()
    if relevance_run_id is None:
        raise ValueError("No relevance run id available for future timeline generation.")
    logging.info("Future timeline relevance_run_id=%s", relevance_run_id)

    if timeline_models is None:
        timeline_models = [
            "openrouter/anthropic/claude-opus-4.5",
            "openrouter/openrouter/bert-nebulon-alpha",
        ]
    if timeline_temps is None:
        timeline_temps = [0.1, 0.3, 0.5, 0.7, 0.9]
    if timeline_rollouts is None:
        timeline_rollouts = 10

    active_event_group_ids = load_active_event_group_ids(active_event_groups_path)
    event_group_index = load_event_group_index(events_path)
    event_groups = [event_group_index[event_group_id] for event_group_id in active_event_group_ids]
    logging.info("Future timeline event_groups=%s", len(event_groups))

    future_service = FutureTimelineService(
        storage=storage,
        timeline_models=timeline_models,
        timeline_temps=timeline_temps,
        timeline_rollouts=timeline_rollouts,
    )
    summaries = future_service.process(
        event_groups,
        relevance_run_id,
        run_id,
        force_without_relevance=force_without_relevance,
    )

    run_finished_at = dt.datetime.now(dt.timezone.utc)
    run_record = {
        "run_id": run_id,
        "started_at": run_started_at.isoformat(),
        "ended_at": run_finished_at.isoformat(),
        "relevance_run_id": relevance_run_id,
        "event_group_summaries": summaries,
        "timeline_models": timeline_models,
        "timeline_temps": timeline_temps,
        "timeline_rollouts": timeline_rollouts,
    }
    storage.append_run_metadata(run_record)
    logging.info("Future timeline run %s completed", run_id)
    return run_record


def run_simulation_pipeline(
    active_event_groups_path: Path,
    events_path: Path,
    news_base_path: Path,
    storage_dir: Path,
    force_present: bool = False,
    target_timeline_chars: int | None = None,
    min_events: int | None = None,
    max_events: int | None = None,
    relevance_model: str = "openrouter/x-ai/grok-4.1-fast",
    timeline_models: list[str] | None = None,
    timeline_temps: list[float] | None = None,
    timeline_rollouts: int | None = None,
    batch_size: int = 25,
) -> dict:
    """Run present timeline, relevance, and future timeline pipelines."""
    logging.info("Simulation pipeline starting")
    present = run_present_timeline_pipeline(
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        storage_dir=storage_dir,
        force_present=force_present,
        target_timeline_chars=target_timeline_chars,
        min_events=min_events,
        max_events=max_events,
    )
    relevance = run_relevance_pipeline(
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        news_base_path=news_base_path,
        storage_dir=storage_dir,
        relevance_model=relevance_model,
        batch_size=batch_size,
    )
    future = run_future_timeline_pipeline(
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        storage_dir=storage_dir,
        relevance_run_id=relevance["run_id"],
        timeline_models=timeline_models,
        timeline_temps=timeline_temps,
        timeline_rollouts=timeline_rollouts,
    )
    return {"present": present, "relevance": relevance, "future": future}
