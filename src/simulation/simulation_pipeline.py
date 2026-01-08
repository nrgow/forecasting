import datetime as dt
import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Iterable

import dspy
import more_itertools as mit

from ..prediction_markets import EventGroup, Events
from .generate_future_timeline import run_model
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

    def __init__(self, models: list[str], temps: list[float]) -> None:
        """Initialize the estimator with model and temperature settings."""
        self.models = models
        self.temps = temps

    def generate(
        self,
        scenario: str,
        contexts: list[str],
        current_date: str,
        final_question: str,
    ) -> dict:
        """Generate future timeline rollouts and implied probability."""
        results = list(
            chain.from_iterable(
                run_model(
                    model=model,
                    scenario=scenario,
                    contexts=contexts,
                    current_date=current_date,
                    final_question=final_question,
                    temps=self.temps,
                )
                for model in self.models
            )
        )
        choices = [result["choice"] for result in results if "choice" in result]
        probability = sum(int(choice) for choice in choices) / len(choices) if choices else None
        return {"results": results, "probability": probability}


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


class RealtimeEstimationService:
    """Evaluate new articles and produce future timelines."""

    def __init__(
        self,
        storage: SimulationStorage,
        relevance_model: str,
        timeline_models: list[str],
        timeline_temps: list[float],
        batch_size: int,
    ) -> None:
        """Initialize the service with models and batch sizing."""
        self.storage = storage
        self.judge = NewsRelevanceJudge(relevance_model)
        self.estimator = FutureTimelineEstimator(timeline_models, timeline_temps)
        self.batch_size = batch_size

    def process(
        self,
        event_groups: list[EventGroup],
        articles: list[NewsArticle],
        run_id: str,
    ) -> list[dict]:
        """Judge relevance, store results, and generate future timelines."""
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
                continue
            relevant_articles = []
            query = build_event_group_prompt(event_group)
            for batch in mit.chunked(candidates, self.batch_size):
                batch_list = list(batch)
                batch_texts = [article.prompt_text() for article in batch_list]
                relevant_texts = set(
                    self.judge.select_relevant(query, batch_texts)
                )
                for article in batch_list:
                    relevant = article.prompt_text() in relevant_texts
                    record = {
                        "event_group_id": event_group_id,
                        "article_id": article.article_id,
                        "article_url": article.url,
                        "article_title": article.title,
                        "article_published_at": article.published_at.isoformat(),
                        "relevant": relevant,
                        "judged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "model": self.judge.model,
                        "run_id": run_id,
                    }
                    self.storage.append_relevance_judgment(record)
                    if relevant:
                        relevant_articles.append(article)
            if not relevant_articles:
                continue
            scenario = build_event_group_prompt(event_group)
            contexts = list(islice((a.prompt_text() for a in relevant_articles), 50))
            timeline_output = self.estimator.generate(
                scenario=scenario,
                contexts=contexts,
                current_date=dt.date.today().isoformat(),
                final_question=event_group.template_title,
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
                "results": timeline_output["results"],
            }
            probability_record = {
                "event_group_id": event_group_id,
                "event_group_title": event_group.template_title,
                "generated_at": generated_at,
                "probability": timeline_output["probability"],
                "samples": len(timeline_output["results"]),
                "run_id": run_id,
            }
            self.storage.append_future_timeline(timeline_record)
            self.storage.append_probability_estimate(probability_record)
            summaries.append(
                {
                    "event_group_id": event_group_id,
                    "relevant_articles": len(relevant_articles),
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

    active_event_group_ids = load_active_event_group_ids(active_event_groups_path)
    event_group_index = load_event_group_index(events_path)
    event_groups = [event_group_index[event_group_id] for event_group_id in active_event_group_ids]

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
    timeline_models: list[str] | None = None,
    timeline_temps: list[float] | None = None,
    batch_size: int = 25,
) -> dict:
    """Run the real-time simulation pipeline for active event groups."""
    storage = SimulationStorage(storage_dir)
    run_id = hashlib.sha256(dt.datetime.now(dt.timezone.utc).isoformat().encode("utf-8")).hexdigest()
    run_started_at = dt.datetime.now(dt.timezone.utc)
    last_run = storage.last_run_metadata()
    if last_run is None:
        news_since = run_started_at - dt.timedelta(hours=24)
    else:
        news_since = dt.datetime.fromisoformat(last_run["news_until"])
    if timeline_models is None:
        timeline_models = [
            "openrouter/anthropic/claude-opus-4.5",
            "openrouter/openrouter/bert-nebulon-alpha",
        ]
    if timeline_temps is None:
        timeline_temps = [0.1, 0.3, 0.5, 0.7, 0.9]

    active_event_group_ids = load_active_event_group_ids(active_event_groups_path)
    event_group_index = load_event_group_index(events_path)
    event_groups = [event_group_index[event_group_id] for event_group_id in active_event_group_ids]

    news_paths = iter_news_paths(news_base_path, news_since)
    articles = list(iter_news_articles(news_paths, news_since))

    realtime_service = RealtimeEstimationService(
        storage=storage,
        relevance_model=relevance_model,
        timeline_models=timeline_models,
        timeline_temps=timeline_temps,
        batch_size=batch_size,
    )
    summaries = realtime_service.process(event_groups, articles, run_id)

    run_finished_at = dt.datetime.now(dt.timezone.utc)
    run_record = {
        "run_id": run_id,
        "started_at": run_started_at.isoformat(),
        "ended_at": run_finished_at.isoformat(),
        "news_since": news_since.isoformat(),
        "news_until": run_finished_at.isoformat(),
        "articles_considered": len(articles),
        "event_group_summaries": summaries,
        "relevance_model": relevance_model,
        "timeline_models": timeline_models,
        "timeline_temps": timeline_temps,
    }
    storage.append_run_metadata(run_record)
    logging.info("Realtime simulation run %s completed", run_id)
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
    batch_size: int = 25,
) -> dict:
    """Run present timeline and real-time pipelines for active event groups."""
    present = run_present_timeline_pipeline(
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        storage_dir=storage_dir,
        force_present=force_present,
        target_timeline_chars=target_timeline_chars,
        min_events=min_events,
        max_events=max_events,
    )
    realtime = run_realtime_simulation_pipeline(
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        news_base_path=news_base_path,
        storage_dir=storage_dir,
        relevance_model=relevance_model,
        timeline_models=timeline_models,
        timeline_temps=timeline_temps,
        batch_size=batch_size,
    )
    return {"present": present, "realtime": realtime}
