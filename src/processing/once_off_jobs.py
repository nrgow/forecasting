import datetime as dt
import hashlib
import logging
import time
from pathlib import Path

import fire

from ..simulation.simulation_pipeline import (
    AgenticRelevanceService,
    BM25NewsIndex,
    DenseNewsIndex,
    PresentTimelineService,
    RelevanceJudgmentService,
    RerankerRelevanceService,
    apply_semantic_deduplication,
    apply_zero_shot_filter,
    iter_news_articles,
    iter_news_paths,
    load_luxical_embedder,
    load_active_event_group_ids,
    load_event_group_index,
    run_present_timeline_pipeline,
)
from ..simulation.storage import SimulationStorage


def run_present_timeline_backfill(
    purge_event_group_id: str = "will-china-invade-taiwan",
    storage_dir: Path = Path("data") / "simulation",
    active_event_groups_path: Path = Path("data") / "active_event_groups.jsonl",
    events_path: Path = Path("data") / "events.jsonl",
    target_timeline_chars: int | None = None,
    min_events: int | None = None,
    max_events: int | None = None,
) -> dict:
    """Purge a present timeline and backfill missing ones for active event groups."""
    logging.info(
        "Once-off present timeline job starting purge_event_group_id=%s",
        purge_event_group_id,
    )
    storage = SimulationStorage(storage_dir)
    purge_started_at = time.perf_counter()
    removed = storage.purge_present_timelines(purge_event_group_id)
    purge_elapsed = time.perf_counter() - purge_started_at
    logging.info(
        "Once-off present timeline purge completed removed=%s seconds=%.2f",
        removed,
        purge_elapsed,
    )

    backfill_started_at = time.perf_counter()
    backfill = run_present_timeline_pipeline(
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        storage_dir=storage_dir,
        force_present=False,
        target_timeline_chars=target_timeline_chars,
        min_events=min_events,
        max_events=max_events,
    )
    backfill_elapsed = time.perf_counter() - backfill_started_at
    logging.info(
        "Once-off present timeline backfill completed seconds=%.2f generated=%s",
        backfill_elapsed,
        backfill["present_timelines_generated"],
    )
    return {"purged": removed, "backfill": backfill}


def run_relevance_backfill_for_event_group(
    event_group_id: str,
    news_base_path: Path,
    storage_dir: Path,
    active_event_groups_path: Path,
    events_path: Path,
    relevance_model: str = "openrouter/x-ai/grok-4.1-fast",
    use_lazy_retriever: bool = True,
    use_reranker: bool = False,
    luxical_model_path: str | None = None,
    luxical_model_id: str | None = "DatologyAI/luxical-one",
    luxical_model_filename: str | None = "luxical_one_rc4.npz",
    news_since_hours: int = 168,
    lazy_search_top_k: int = 25,
    lazy_max_iters: int = 6,
    lazy_max_results: int = 50,
    lazy_dedup_similarity_threshold: float = 0.97,
    reranker_model_name: str = "Qwen/Qwen3-Reranker-0.6B",
    reranker_backend: str = "auto",
    reranker_instruction: str | None = None,
    reranker_max_length: int = 8192,
    reranker_batch_size: int = 8,
    reranker_min_score: float = 0.5,
    zero_shot_min_probability: float = 0.5,
    zero_shot_class_name: str = "international politics geopolitics world financial markets",
    dedup_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dedup_similarity_threshold: float = 0.92,
    dedup_batch_size: int = 128,
    batch_size: int = 25,
) -> dict:
    """Run relevance judgments for a single event group with a custom lookback window."""
    storage = SimulationStorage(storage_dir)
    run_id = hashlib.sha256(
        dt.datetime.now(dt.timezone.utc).isoformat().encode("utf-8")
    ).hexdigest()
    run_started_at = dt.datetime.now(dt.timezone.utc)
    logging.info(
        "Once-off relevance backfill starting event_group_id=%s run_id=%s",
        event_group_id,
        run_id,
    )
    if use_reranker and reranker_instruction is None:
        reranker_instruction = "Given an event description, determine whether the news article is relevant."
    active_event_group_ids = load_active_event_group_ids(active_event_groups_path)
    if event_group_id not in active_event_group_ids:
        raise ValueError(
            f"Event group {event_group_id} not in active list {active_event_groups_path}."
        )
    event_group_index = load_event_group_index(events_path)
    event_group = event_group_index[event_group_id]

    news_since = run_started_at - dt.timedelta(hours=news_since_hours)
    logging.info(
        "Once-off relevance backfill loading news news_since=%s base_path=%s",
        news_since.isoformat(),
        news_base_path,
    )
    load_started_at = time.perf_counter()
    news_paths = iter_news_paths(news_base_path, news_since)
    articles = list(iter_news_articles(news_paths, news_since))
    load_elapsed = time.perf_counter() - load_started_at
    logging.info(
        "Once-off relevance backfill loaded articles=%s news_paths=%s seconds=%.2f",
        len(articles),
        len(news_paths),
        load_elapsed,
    )

    if use_lazy_retriever:
        logging.info(
            "Once-off relevance backfill lazy retrieval starting event_group_id=%s",
            event_group_id,
        )
        if luxical_model_path is None and luxical_model_id is None:
            raise ValueError(
                "luxical_model_path or luxical_model_id is required for dense retrieval."
            )
        embedder, resolved_model_path = load_luxical_embedder(
            luxical_model_path,
            luxical_model_id,
            luxical_model_filename,
        )
        bm25_started_at = time.perf_counter()
        bm25_index = BM25NewsIndex(articles)
        bm25_elapsed = time.perf_counter() - bm25_started_at
        logging.info(
            "Once-off relevance backfill BM25 ready articles=%s seconds=%.2f",
            len(articles),
            bm25_elapsed,
        )
        dense_started_at = time.perf_counter()
        dense_index = DenseNewsIndex(
            articles,
            luxical_model_path=resolved_model_path,
            luxical_model_id=luxical_model_id,
            luxical_model_filename=luxical_model_filename,
            embedder=embedder,
        )
        dense_elapsed = time.perf_counter() - dense_started_at
        logging.info(
            "Once-off relevance backfill dense ready articles=%s seconds=%.2f",
            len(articles),
            dense_elapsed,
        )
        present_timeline_index = storage.load_present_timeline_index()
        relevance_service = AgenticRelevanceService(
            storage=storage,
            relevance_model=relevance_model,
            search_top_k=lazy_search_top_k,
            max_iters=lazy_max_iters,
            max_results=lazy_max_results,
            dedup_embedder=embedder,
            dedup_similarity_threshold=lazy_dedup_similarity_threshold,
        )
        relevance_started_at = time.perf_counter()
        summaries, total_candidates = relevance_service.process(
            event_groups=[event_group],
            bm25_index=bm25_index,
            dense_index=dense_index,
            present_timeline_index=present_timeline_index,
            run_id=run_id,
        )
        relevance_elapsed = time.perf_counter() - relevance_started_at
        articles_considered = total_candidates
        clusters_considered = None
        dedup_ratio = None
        logging.info(
            "Once-off relevance backfill lazy completed seconds=%.2f candidates=%s",
            relevance_elapsed,
            total_candidates,
        )
    else:
        logging.info(
            "Once-off relevance backfill eager starting event_group_id=%s",
            event_group_id,
        )
        zero_shot_started_at = time.perf_counter()
        if use_reranker:
            logging.info(
                "Once-off relevance backfill reranker starting event_group_id=%s reranker_model_name=%s",
                event_group_id,
                reranker_model_name,
            )
            articles = apply_zero_shot_filter(
                articles=articles,
                storage=storage,
                run_id=run_id,
                class_name=zero_shot_class_name,
                min_probability=zero_shot_min_probability,
            )
            zero_shot_elapsed = time.perf_counter() - zero_shot_started_at
            logging.info(
                "Once-off relevance backfill zero-shot completed seconds=%.2f articles=%s",
                zero_shot_elapsed,
                len(articles),
            )
            relevance_service = RerankerRelevanceService(
                storage=storage,
                model_name=reranker_model_name,
                backend=reranker_backend,
                instruction=reranker_instruction,
                max_length=reranker_max_length,
                batch_size=reranker_batch_size,
                min_score=reranker_min_score,
            )
            relevance_started_at = time.perf_counter()
            summaries = relevance_service.process([event_group], articles, run_id)
            relevance_elapsed = time.perf_counter() - relevance_started_at
            articles_considered = len(articles)
            clusters_considered = None
            dedup_ratio = None
            logging.info(
                "Once-off relevance backfill reranker completed seconds=%.2f articles=%s",
                relevance_elapsed,
                articles_considered,
            )
        else:
            articles = apply_zero_shot_filter(
                articles=articles,
                storage=storage,
                run_id=run_id,
                class_name=zero_shot_class_name,
                min_probability=zero_shot_min_probability,
            )
            zero_shot_elapsed = time.perf_counter() - zero_shot_started_at
            logging.info(
                "Once-off relevance backfill zero-shot completed seconds=%.2f articles=%s",
                zero_shot_elapsed,
                len(articles),
            )
            dedup_started_at = time.perf_counter()
            dedup_clusters = apply_semantic_deduplication(
                articles=articles,
                model_name=dedup_model_name,
                similarity_threshold=dedup_similarity_threshold,
                batch_size=dedup_batch_size,
            )
            dedup_elapsed = time.perf_counter() - dedup_started_at
            logging.info(
                "Once-off relevance backfill dedup completed seconds=%.2f clusters=%s",
                dedup_elapsed,
                len(dedup_clusters),
            )
            dedup_ratio = len(dedup_clusters) / len(articles) if articles else 0.0
            relevance_service = RelevanceJudgmentService(
                storage=storage,
                relevance_model=relevance_model,
                batch_size=batch_size,
            )
            relevance_started_at = time.perf_counter()
            summaries = relevance_service.process([event_group], dedup_clusters, run_id)
            relevance_elapsed = time.perf_counter() - relevance_started_at
            articles_considered = len(articles)
            clusters_considered = len(dedup_clusters)
            logging.info(
                "Once-off relevance backfill eager completed seconds=%.2f articles=%s clusters=%s",
                relevance_elapsed,
                articles_considered,
                clusters_considered,
            )

    run_finished_at = dt.datetime.now(dt.timezone.utc)
    run_record = {
        "run_id": run_id,
        "started_at": run_started_at.isoformat(),
        "ended_at": run_finished_at.isoformat(),
        "news_since": news_since.isoformat(),
        "news_until": run_finished_at.isoformat(),
        "articles_total": len(articles),
        "articles_considered": articles_considered,
        "clusters_considered": clusters_considered,
        "dedup_ratio": dedup_ratio,
        "event_group_summaries": summaries,
        "relevance_model": relevance_model,
        "retrieval_strategy": "lazy" if use_lazy_retriever else "eager",
        "relevance_mode": (
            "lazy"
            if use_lazy_retriever
            else "eager_reranker" if use_reranker else "eager_llm"
        ),
        "bm25_indexed_articles": len(articles) if use_lazy_retriever else None,
        "dense_indexed_articles": len(articles) if use_lazy_retriever else None,
        "bm25_candidates": articles_considered if use_lazy_retriever else None,
        "lazy_search_top_k": lazy_search_top_k if use_lazy_retriever else None,
        "lazy_max_iters": lazy_max_iters if use_lazy_retriever else None,
        "lazy_max_results": lazy_max_results if use_lazy_retriever else None,
        "zero_shot_class_name": (
            zero_shot_class_name if not use_lazy_retriever else None
        ),
        "zero_shot_min_probability": (
            zero_shot_min_probability if not use_lazy_retriever else None
        ),
        "dedup_model_name": dedup_model_name if not use_lazy_retriever else None,
        "dedup_similarity_threshold": (
            dedup_similarity_threshold if not use_lazy_retriever else None
        ),
        "dedup_batch_size": dedup_batch_size if not use_lazy_retriever else None,
        "reranker_model_name": reranker_model_name if use_reranker else None,
        "reranker_backend": reranker_backend if use_reranker else None,
        "reranker_instruction": reranker_instruction if use_reranker else None,
        "reranker_max_length": reranker_max_length if use_reranker else None,
        "reranker_batch_size": reranker_batch_size if use_reranker else None,
        "reranker_min_score": reranker_min_score if use_reranker else None,
    }
    storage.append_run_metadata(run_record)
    logging.info(
        "Once-off relevance backfill completed event_group_id=%s run_id=%s",
        event_group_id,
        run_id,
    )
    return run_record


def run_russia_ukraine_ceasefire_backfill(
    event_group_id: str = "russia-x-ukraine-ceasefire",
    storage_dir: Path = Path("data") / "simulation",
    active_event_groups_path: Path = Path("data") / "active_event_groups.jsonl",
    events_path: Path = Path("data") / "events.jsonl",
    news_base_path: Path = Path("/mnt/ssd") / "newstalk-data" / "gdelt-gal",
    force_present: bool = True,
    target_timeline_chars: int | None = None,
    min_events: int | None = None,
    max_events: int | None = None,
    relevance_model: str = "openrouter/x-ai/grok-4.1-fast",
    use_lazy_retriever: bool = True,
    use_reranker: bool = False,
    luxical_model_path: str | None = None,
    luxical_model_id: str | None = "DatologyAI/luxical-one",
    luxical_model_filename: str | None = "luxical_one_rc4.npz",
    news_since_hours: int = 168,
) -> dict:
    """Backfill present timeline and relevance judgments for the Russia-Ukraine ceasefire."""
    logging.info(
        "Once-off backfill starting event_group_id=%s news_since_hours=%s",
        event_group_id,
        news_since_hours,
    )
    storage = SimulationStorage(storage_dir)
    purge_started_at = time.perf_counter()
    removed = storage.purge_present_timelines(event_group_id)
    purge_elapsed = time.perf_counter() - purge_started_at
    logging.info(
        "Once-off present timeline purge completed removed=%s seconds=%.2f",
        removed,
        purge_elapsed,
    )

    active_event_group_ids = load_active_event_group_ids(active_event_groups_path)
    if event_group_id not in active_event_group_ids:
        raise ValueError(
            f"Event group {event_group_id} not in active list {active_event_groups_path}."
        )
    event_group_index = load_event_group_index(events_path)
    event_group = event_group_index[event_group_id]

    present_started_at = time.perf_counter()
    present_service = PresentTimelineService(storage)
    present_records = present_service.generate_if_missing(
        [event_group],
        force_present,
        target_timeline_chars,
        min_events,
        max_events,
    )
    present_elapsed = time.perf_counter() - present_started_at
    logging.info(
        "Once-off present timeline backfill completed seconds=%.2f generated=%s",
        present_elapsed,
        len(present_records),
    )

    relevance_record = run_relevance_backfill_for_event_group(
        event_group_id=event_group_id,
        news_base_path=news_base_path,
        storage_dir=storage_dir,
        active_event_groups_path=active_event_groups_path,
        events_path=events_path,
        relevance_model=relevance_model,
        use_lazy_retriever=use_lazy_retriever,
        use_reranker=use_reranker,
        luxical_model_path=luxical_model_path,
        luxical_model_id=luxical_model_id,
        luxical_model_filename=luxical_model_filename,
        news_since_hours=news_since_hours,
    )
    return {
        "purged_present": removed,
        "present_backfill": present_records,
        "relevance_backfill": relevance_record,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire()
