import datetime as dt
import gzip
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Callable, Iterable

import dspy
import more_itertools as mit
import numpy as np
from huggingface_hub import hf_hub_download
from luxical.embedder import Embedder
from sentence_transformers import SentenceTransformer

from ..prediction_markets import EventGroup, Events, parse_open_market_end_date
from ..news_enrichment import ZeroShotClassifier
from .generate_future_timeline import run_model, sanitize_future_timeline_topic
from .generate_present_timeline import generate_present_timeline
from .bm25_retriever import BM25NewsIndex, BM25SearchTool
from .dense_retriever import DenseNewsIndex, DenseSearchTool, DualSearchAgent
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


@dataclass
class ArticleCluster:
    """Cluster of semantically near-duplicate articles."""

    cluster_id: str
    representative: NewsArticle
    members: list[NewsArticle]


class SemanticDeduplicator:
    """Cluster near-duplicate articles using sentence embeddings."""

    def __init__(
        self,
        model_name: str,
        similarity_threshold: float,
        batch_size: int,
    ) -> None:
        """Initialize the deduplicator with model and similarity settings."""
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.model_name, device="cuda")

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return normalized embeddings for the provided texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings)

    def cluster(self, articles: list[NewsArticle]) -> list[ArticleCluster]:
        """Return clusters of articles based on cosine similarity."""
        if not articles:
            return []
        texts = [article.prompt_text() for article in articles]
        embeddings = self._embed(texts)
        assigned = np.zeros(len(articles), dtype=bool)
        clusters: list[ArticleCluster] = []
        for idx in range(len(articles)):
            if assigned[idx]:
                continue
            similarities = embeddings @ embeddings[idx]
            members_idx = np.where(similarities >= self.similarity_threshold)[0].tolist()
            members_idx = [member_idx for member_idx in members_idx if not assigned[member_idx]]
            for member_idx in members_idx:
                assigned[member_idx] = True
            representative_idx = max(
                members_idx, key=lambda member_idx: articles[member_idx].published_at
            )
            members = [articles[member_idx] for member_idx in members_idx]
            cluster_key = "|".join(sorted(member.article_id for member in members))
            cluster_id = hashlib.sha256(cluster_key.encode("utf-8")).hexdigest()
            clusters.append(
                ArticleCluster(
                    cluster_id=cluster_id,
                    representative=articles[representative_idx],
                    members=members,
                )
            )
        return clusters


def load_luxical_embedder(
    luxical_model_path: str | None,
    luxical_model_id: str | None,
    luxical_model_filename: str | None,
) -> tuple[Embedder, str]:
    """Load a Luxical embedder for deduplication and dense retrieval."""
    if luxical_model_path is None:
        if luxical_model_id is None or luxical_model_filename is None:
            raise ValueError(
                "luxical_model_path or luxical_model_id + luxical_model_filename is required."
            )
        logging.info(
            "Luxical embedder download starting repo_id=%s filename=%s",
            luxical_model_id,
            luxical_model_filename,
        )
        started_at = time.perf_counter()
        luxical_model_path = hf_hub_download(
            repo_id=luxical_model_id,
            filename=luxical_model_filename,
        )
        elapsed = time.perf_counter() - started_at
        logging.info(
            "Luxical embedder download completed repo_id=%s filename=%s seconds=%.2f",
            luxical_model_id,
            luxical_model_filename,
            elapsed,
        )
    logging.info("Luxical embedder load starting path=%s", luxical_model_path)
    started_at = time.perf_counter()
    embedder = Embedder.load(luxical_model_path)
    elapsed = time.perf_counter() - started_at
    logging.info("Luxical embedder load completed seconds=%.2f", elapsed)
    return embedder, luxical_model_path


def _normalize_dedup_text(text: str, strip_punct: bool) -> str:
    """Return a normalized string for deduplication keys."""
    normalized = text.strip()
    if strip_punct:
        normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower()


def _dedup_by_key(
    articles: list[NewsArticle],
    key_fn: Callable[[NewsArticle], str],
) -> list[NewsArticle]:
    """Return articles deduplicated by a key function."""
    seen: set[str] = set()
    unique: list[NewsArticle] = []
    for article in articles:
        key = key_fn(article)
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    return unique


def _dedup_semantic(
    articles: list[NewsArticle],
    embedder: Embedder,
    similarity_threshold: float,
) -> list[NewsArticle]:
    """Return articles deduplicated by Luxical semantic similarity."""
    if not articles:
        return []
    logging.info(
        "Semantic deduplication starting articles=%s threshold=%.3f",
        len(articles),
        similarity_threshold,
    )
    started_at = time.perf_counter()
    texts = [article.prompt_text() for article in articles]
    embeddings = np.asarray(embedder(texts, progress_bars=False), dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    kept_indices: list[int] = []
    kept_embeddings: list[np.ndarray] = []
    for idx, embedding in enumerate(embeddings):
        if kept_embeddings:
            similarities = np.dot(np.asarray(kept_embeddings), embedding)
            if similarities.max() >= similarity_threshold:
                continue
        kept_indices.append(idx)
        kept_embeddings.append(embedding)
    unique_articles = [articles[idx] for idx in kept_indices]
    elapsed = time.perf_counter() - started_at
    logging.info(
        "Semantic deduplication completed kept=%s removed=%s seconds=%.2f",
        len(unique_articles),
        len(articles) - len(unique_articles),
        elapsed,
    )
    return unique_articles


def _cluster_semantic(
    articles: list[NewsArticle],
    embedder: Embedder,
    similarity_threshold: float,
) -> list[ArticleCluster]:
    """Return clusters of articles using Luxical semantic similarity."""
    if not articles:
        return []
    texts = [article.prompt_text() for article in articles]
    embeddings = np.asarray(embedder(texts, progress_bars=False), dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    assigned = np.zeros(len(articles), dtype=bool)
    clusters: list[ArticleCluster] = []
    for idx in range(len(articles)):
        if assigned[idx]:
            continue
        similarities = embeddings @ embeddings[idx]
        members_idx = np.where(similarities >= similarity_threshold)[0].tolist()
        members_idx = [member_idx for member_idx in members_idx if not assigned[member_idx]]
        for member_idx in members_idx:
            assigned[member_idx] = True
        representative_idx = max(
            members_idx, key=lambda member_idx: articles[member_idx].published_at
        )
        members = [articles[member_idx] for member_idx in members_idx]
        cluster_key = "|".join(sorted(member.article_id for member in members))
        cluster_id = hashlib.sha256(cluster_key.encode("utf-8")).hexdigest()
        clusters.append(
            ArticleCluster(
                cluster_id=cluster_id,
                representative=articles[representative_idx],
                members=members,
            )
        )
    return clusters


def deduplicate_articles_for_retrieval(
    articles: list[NewsArticle],
    embedder: Embedder,
    similarity_threshold: float,
) -> list[NewsArticle]:
    """Deduplicate articles using exact, punctuation-stripped, and semantic steps."""
    logging.info("Retrieval deduplication starting articles=%s", len(articles))
    started_at = time.perf_counter()
    ordered_articles = sorted(
        articles, key=lambda article: article.published_at, reverse=True
    )

    exact_started_at = time.perf_counter()
    exact_unique = _dedup_by_key(ordered_articles, lambda article: article.prompt_text())
    exact_elapsed = time.perf_counter() - exact_started_at
    logging.info(
        "Retrieval deduplication exact completed kept=%s removed=%s seconds=%.2f",
        len(exact_unique),
        len(ordered_articles) - len(exact_unique),
        exact_elapsed,
    )

    punct_started_at = time.perf_counter()
    punct_unique = _dedup_by_key(
        exact_unique,
        lambda article: _normalize_dedup_text(article.prompt_text(), strip_punct=True),
    )
    punct_elapsed = time.perf_counter() - punct_started_at
    logging.info(
        "Retrieval deduplication punctuation completed kept=%s removed=%s seconds=%.2f",
        len(punct_unique),
        len(exact_unique) - len(punct_unique),
        punct_elapsed,
    )

    semantic_unique = _dedup_semantic(
        punct_unique,
        embedder,
        similarity_threshold,
    )
    total_elapsed = time.perf_counter() - started_at
    logging.info(
        "Retrieval deduplication completed kept=%s removed=%s seconds=%.2f",
        len(semantic_unique),
        len(articles) - len(semantic_unique),
        total_elapsed,
    )
    return semantic_unique


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
        if result is None:
            raise ValueError(
                f"Relevance model returned no result for model={self.model} articles={len(articles)}"
            )
        if result.relevant_articles is None:
            raise ValueError(
                f"Relevance model returned no relevant_articles for model={self.model} articles={len(articles)}"
            )
        if not isinstance(result.relevant_articles, list):
            raise ValueError(
                "Relevance model returned non-list relevant_articles "
                f"type={type(result.relevant_articles)} model={self.model}"
            )
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
            record = {
                "event_group_id": event_group.id(),
                "event_group_title": event_group.template_title,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "timeline": output,
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
        clusters: list[ArticleCluster],
        run_id: str,
    ) -> list[dict]:
        """Judge relevance and store results."""
        logging.info(
            "Relevance judgment starting event_groups=%s clusters=%s batch_size=%s",
            len(event_groups),
            len(clusters),
            self.batch_size,
        )
        existing = self.storage.load_relevance_index()
        summaries = []
        for event_group in event_groups:
            event_group_id = event_group.id()
            candidates = []
            for cluster in clusters:
                if any(
                    (event_group_id, member.article_id) not in existing
                    for member in cluster.members
                ):
                    candidates.append(cluster)
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
            considered_articles = 0
            query = build_event_group_prompt(event_group)
            for batch_index, batch in enumerate(
                mit.chunked(candidates, self.batch_size), start=1
            ):
                batch_list = list(batch)
                batch_texts = [
                    cluster.representative.prompt_text() for cluster in batch_list
                ]
                started_at = time.perf_counter()
                relevant_texts = set(self.judge.select_relevant(query, batch_texts))
                elapsed = time.perf_counter() - started_at
                items_per_sec = len(batch_list) / elapsed
                relevant_in_batch = 0
                for cluster in batch_list:
                    relevant = cluster.representative.prompt_text() in relevant_texts
                    for member in cluster.members:
                        if (event_group_id, member.article_id) in existing:
                            continue
                        record = {
                            "event_group_id": event_group_id,
                            "article_id": member.article_id,
                            "article_url": member.url,
                            "article_title": member.title,
                            "article_desc": member.desc,
                            "article_published_at": member.published_at.isoformat(),
                            "relevant": relevant,
                            "judged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "model": self.judge.model,
                            "run_id": run_id,
                            "dedup_cluster_id": cluster.cluster_id,
                            "dedup_cluster_size": len(cluster.members),
                            "dedup_representative_id": cluster.representative.article_id,
                            "dedup_representative": member.article_id
                            == cluster.representative.article_id,
                        }
                        self.storage.append_relevance_judgment(record)
                        considered_articles += 1
                        if relevant:
                            relevant_in_batch += 1
                            relevant_articles += 1
                logging.info(
                    "Relevance batch event_group_id=%s batch=%s/%s batch_size=%s relevant_articles=%s",
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
                    "clusters_considered": len(candidates),
                    "articles_considered": considered_articles,
                    "relevant_articles": relevant_articles,
                }
            )
        return summaries


class AgenticRelevanceService:
    """Agentic dense + BM25 retrieval and relevance selection for event groups."""

    def __init__(
        self,
        storage: SimulationStorage,
        relevance_model: str,
        search_top_k: int,
        max_iters: int,
        max_results: int,
        dedup_embedder: Embedder,
        dedup_similarity_threshold: float,
    ) -> None:
        """Initialize the service with dense + BM25 retrieval settings."""
        self.storage = storage
        self.relevance_model = relevance_model
        self.search_top_k = search_top_k
        self.max_iters = max_iters
        self.max_results = max_results
        self.dedup_embedder = dedup_embedder
        self.dedup_similarity_threshold = dedup_similarity_threshold

    def process(
        self,
        event_groups: list[EventGroup],
        bm25_index: BM25NewsIndex,
        dense_index: DenseNewsIndex,
        present_timeline_index: dict[str, dict],
        run_id: str,
    ) -> tuple[list[dict], int]:
        """Retrieve and store agent-selected relevance records."""
        summaries = []
        total_candidates = 0
        for event_group in event_groups:
            event_group_id = event_group.id()
            logging.info(
                "Relevance lazy agent starting event_group_id=%s max_iters=%s max_results=%s search_top_k=%s",
                event_group_id,
                self.max_iters,
                self.max_results,
                self.search_top_k,
            )
            present_record = present_timeline_index[event_group_id]
            present_timeline = build_present_timeline_context(present_record)
            bm25_tool = BM25SearchTool(
                bm25_index,
                default_top_k=self.search_top_k,
                event_group_id=event_group_id,
            )
            dense_tool = DenseSearchTool(
                dense_index,
                default_top_k=self.search_top_k,
                event_group_id=event_group_id,
            )
            agent = DualSearchAgent(
                model=self.relevance_model,
                bm25_tool=bm25_tool,
                dense_tool=dense_tool,
                max_iters=self.max_iters,
            )
            event_prompt = build_event_group_prompt(event_group)
            logging.info(
                "Deep search agent querying event_group_id=%s",
                event_group_id,
            )
            search_started_at = time.perf_counter()
            relevant_article_ids = agent.find_relevant(
                event_group_prompt=event_prompt,
                present_timeline=present_timeline,
                max_results=self.max_results,
            )
            candidate_by_id: dict[str, NewsArticle] = {}
            for article in bm25_tool.collected_articles():
                candidate_by_id[article.article_id] = article
            for article in dense_tool.collected_articles():
                candidate_by_id[article.article_id] = article
            candidate_articles = list(candidate_by_id.values())
            candidate_ids = set(candidate_by_id.keys())
            elapsed = time.perf_counter() - search_started_at
            logging.info(
                "Deep search agent completed event_group_id=%s seconds=%.2f relevant_ids=%s bm25_queries=%s dense_queries=%s candidates=%s",
                event_group_id,
                elapsed,
                len(relevant_article_ids),
                len(bm25_tool.query_log),
                len(dense_tool.query_log),
                len(candidate_articles),
            )
            unknown_ids = [article_id for article_id in relevant_article_ids if article_id not in candidate_ids]
            if unknown_ids:
                logging.warning(
                    "Deep search agent returned unknown article ids event_group_id=%s unknown_ids=%s",
                    event_group_id,
                    unknown_ids,
                )
                relevant_article_ids = [
                    article_id for article_id in relevant_article_ids if article_id in candidate_ids
                ]
            relevant_set = set(relevant_article_ids)
            logging.info(
                "Deep search semantic deduplication starting event_group_id=%s candidates=%s threshold=%.3f",
                event_group_id,
                len(candidate_articles),
                self.dedup_similarity_threshold,
            )
            dedup_started_at = time.perf_counter()
            clusters = _cluster_semantic(
                candidate_articles,
                self.dedup_embedder,
                self.dedup_similarity_threshold,
            )
            dedup_elapsed = time.perf_counter() - dedup_started_at
            logging.info(
                "Deep search semantic deduplication completed event_group_id=%s clusters=%s candidates=%s seconds=%.2f",
                event_group_id,
                len(clusters),
                len(candidate_articles),
                dedup_elapsed,
            )
            relevant_articles = 0
            for cluster in clusters:
                relevant = any(
                    member.article_id in relevant_set for member in cluster.members
                )
                for member in cluster.members:
                    record = {
                        "event_group_id": event_group_id,
                        "article_id": member.article_id,
                        "article_url": member.url,
                        "article_title": member.title,
                        "article_desc": member.desc,
                        "article_published_at": member.published_at.isoformat(),
                        "relevant": relevant,
                        "judged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "model": self.relevance_model,
                        "run_id": run_id,
                        "dedup_cluster_id": cluster.cluster_id,
                        "dedup_cluster_size": len(cluster.members),
                        "dedup_representative_id": cluster.representative.article_id,
                        "dedup_representative": member.article_id
                        == cluster.representative.article_id,
                        "retrieval_strategy": "lazy",
                    }
                    self.storage.append_relevance_judgment(record)
                    if relevant:
                        relevant_articles += 1
            total_candidates += len(candidate_articles)
            summaries.append(
                {
                    "event_group_id": event_group_id,
                    "articles_considered": len(candidate_articles),
                    "relevant_articles": relevant_articles,
                    "bm25_queries": len(bm25_tool.query_log),
                    "dense_queries": len(dense_tool.query_log),
                }
            )
        return summaries, total_candidates


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


def build_present_timeline_context(record: dict) -> str:
    """Return the present timeline text for an event group record."""
    return record["timeline"]["merged"]["merged_timeline"]


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


def apply_semantic_deduplication(
    articles: list[NewsArticle],
    model_name: str,
    similarity_threshold: float,
    batch_size: int,
) -> list[ArticleCluster]:
    """Cluster near-duplicate articles to reduce relevance calls."""
    deduplicator = SemanticDeduplicator(
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size,
    )
    return deduplicator.cluster(articles)


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
    use_lazy_retriever: bool = False,
    luxical_model_path: str | None = None,
    luxical_model_id: str | None = "DatologyAI/luxical-one",
    luxical_model_filename: str | None = "luxical_one_rc4.npz",
    lazy_search_top_k: int = 25,
    lazy_max_iters: int = 6,
    lazy_max_results: int = 50,
    timeline_models: list[str] | None = ["openrouter/x-ai/grok-4.1-fast"],
    timeline_temps: list[float] | None = [0.7],
    batch_size: int = 25,
) -> dict:
    """Run the real-time simulation pipeline for active event groups."""
    logging.info("Real-time simulation pipeline starting")
    logging.info(
        "Real-time relevance mode=%s",
        "lazy(agentic)" if use_lazy_retriever else "eager(classifier)",
    )
    relevance = run_relevance_pipeline(
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        news_base_path=news_base_path,
        storage_dir=storage_dir,
        relevance_model=relevance_model,
        use_lazy_retriever=use_lazy_retriever,
        luxical_model_path=luxical_model_path,
        luxical_model_id=luxical_model_id,
        luxical_model_filename=luxical_model_filename,
        lazy_search_top_k=lazy_search_top_k,
        lazy_max_iters=lazy_max_iters,
        lazy_max_results=lazy_max_results,
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
    use_lazy_retriever: bool = False,
    luxical_model_path: str | None = None,
    luxical_model_id: str | None = "DatologyAI/luxical-one",
    luxical_model_filename: str | None = "luxical_one_rc4.npz",
    lazy_search_top_k: int = 25,
    lazy_max_iters: int = 6,
    lazy_max_results: int = 50,
    lazy_dedup_similarity_threshold: float = 0.97,
    zero_shot_min_probability: float = 0.5,
    zero_shot_class_name: str = "international politics geopolitics world financial markets",
    dedup_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dedup_similarity_threshold: float = 0.92,
    dedup_batch_size: int = 128,
    batch_size: int = 25,
) -> dict:
    """Run the relevance judgment pipeline for active event groups."""
    storage = SimulationStorage(storage_dir)
    run_id = hashlib.sha256(dt.datetime.now(dt.timezone.utc).isoformat().encode("utf-8")).hexdigest()
    run_started_at = dt.datetime.now(dt.timezone.utc)
    logging.info("Relevance run %s starting", run_id)
    logging.info(
        "Relevance module=%s",
        "lazy(agentic)" if use_lazy_retriever else "eager(classifier)",
    )
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
    if use_lazy_retriever:
        logging.info(
            "Relevance lazy retrieval articles=%s news_paths=%s",
            len(articles),
            len(news_paths),
        )
        present_timeline_index = storage.load_present_timeline_index()
        if luxical_model_path is None and luxical_model_id is None:
            raise ValueError("luxical_model_path or luxical_model_id is required for dense retrieval.")
        embedder, resolved_model_path = load_luxical_embedder(
            luxical_model_path,
            luxical_model_id,
            luxical_model_filename,
        )
        logging.info(
            "Relevance lazy building BM25 index articles=%s",
            len(articles),
        )
        bm25_index = BM25NewsIndex(articles)
        logging.info(
            "Relevance lazy building dense index articles=%s",
            len(articles),
        )
        dense_index = DenseNewsIndex(
            articles,
            luxical_model_path=resolved_model_path,
            luxical_model_id=luxical_model_id,
            luxical_model_filename=luxical_model_filename,
            embedder=embedder,
        )
        relevance_service = AgenticRelevanceService(
            storage=storage,
            relevance_model=relevance_model,
            search_top_k=lazy_search_top_k,
            max_iters=lazy_max_iters,
            max_results=lazy_max_results,
            dedup_embedder=embedder,
            dedup_similarity_threshold=lazy_dedup_similarity_threshold,
        )
        summaries, total_candidates = relevance_service.process(
            event_groups=event_groups,
            bm25_index=bm25_index,
            dense_index=dense_index,
            present_timeline_index=present_timeline_index,
            run_id=run_id,
        )
        articles_considered = total_candidates
        clusters_considered = None
        dedup_ratio = None
    else:
        logging.info(
            "Relevance eager pipeline starting zero_shot_class_name=%s dedup_model_name=%s",
            zero_shot_class_name,
            dedup_model_name,
        )
        articles = apply_zero_shot_filter(
            articles=articles,
            storage=storage,
            run_id=run_id,
            class_name=zero_shot_class_name,
            min_probability=zero_shot_min_probability,
        )
        dedup_clusters = apply_semantic_deduplication(
            articles=articles,
            model_name=dedup_model_name,
            similarity_threshold=dedup_similarity_threshold,
            batch_size=dedup_batch_size,
        )
        dedup_ratio = len(dedup_clusters) / len(articles) if articles else 0.0
        logging.info(
            "Relevance articles=%s clusters=%s news_paths=%s batch_size=%s",
            len(articles),
            len(dedup_clusters),
            len(news_paths),
            batch_size,
        )

        relevance_service = RelevanceJudgmentService(
            storage=storage,
            relevance_model=relevance_model,
            batch_size=batch_size,
        )
        summaries = relevance_service.process(event_groups, dedup_clusters, run_id)
        articles_considered = len(articles)
        clusters_considered = len(dedup_clusters)

    run_finished_at = dt.datetime.now(dt.timezone.utc)
    run_record = {
        "run_id": run_id,
        "started_at": run_started_at.isoformat(),
        "ended_at": run_finished_at.isoformat(),
        "news_since": news_since.isoformat(),
        "news_until": run_finished_at.isoformat(),
        "articles_total": articles_total,
        "articles_considered": articles_considered,
        "clusters_considered": clusters_considered,
        "dedup_ratio": dedup_ratio,
        "event_group_summaries": summaries,
        "relevance_model": relevance_model,
        "retrieval_strategy": "lazy" if use_lazy_retriever else "eager",
        "bm25_indexed_articles": articles_total if use_lazy_retriever else None,
        "dense_indexed_articles": articles_total if use_lazy_retriever else None,
        "bm25_candidates": articles_considered if use_lazy_retriever else None,
        "lazy_search_top_k": lazy_search_top_k if use_lazy_retriever else None,
        "lazy_max_iters": lazy_max_iters if use_lazy_retriever else None,
        "lazy_max_results": lazy_max_results if use_lazy_retriever else None,
        "zero_shot_class_name": zero_shot_class_name if not use_lazy_retriever else None,
        "zero_shot_min_probability": zero_shot_min_probability if not use_lazy_retriever else None,
        "dedup_model_name": dedup_model_name if not use_lazy_retriever else None,
        "dedup_similarity_threshold": dedup_similarity_threshold if not use_lazy_retriever else None,
        "dedup_batch_size": dedup_batch_size if not use_lazy_retriever else None,
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
    use_lazy_retriever: bool = False,
    luxical_model_path: str | None = None,
    luxical_model_id: str | None = "DatologyAI/luxical-one",
    luxical_model_filename: str | None = "luxical_one_rc4.npz",
    lazy_search_top_k: int = 25,
    lazy_max_iters: int = 6,
    lazy_max_results: int = 50,
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
        use_lazy_retriever=use_lazy_retriever,
        luxical_model_path=luxical_model_path,
        luxical_model_id=luxical_model_id,
        luxical_model_filename=luxical_model_filename,
        lazy_search_top_k=lazy_search_top_k,
        lazy_max_iters=lazy_max_iters,
        lazy_max_results=lazy_max_results,
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
