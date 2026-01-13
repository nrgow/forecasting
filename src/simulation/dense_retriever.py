from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import json
import logging
from pathlib import Path
import pickle
import sys
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import hf_hub_download

#import leann_backend_hnsw.hnsw_backend  # noqa: E402
from leann.api import BACKEND_REGISTRY, LeannBuilder, PassageManager  # noqa: E402
from leann.interface import LeannBackendSearcherInterface  # noqa: E402
from luxical.embedder import Embedder  # noqa: E402

import dspy  # noqa: E402

from .bm25_retriever import BM25SearchTool  # noqa: E402
from .news_relevance_dspy import DeepSearchRelevantArticles  # noqa: E402

if TYPE_CHECKING:
    from .simulation_pipeline import NewsArticle

DEEP_SEARCH_INSTRUCTIONS = (
    "You are searching a corpus of very recent news headlines (last 24-72 hours). "
    "Generate multiple diverse, on-topic queries across distinct angles, entities, "
    "and causal frames. Before calling tools, outline a brief query plan with 3-5 "
    "distinct angles, then execute queries that reflect that plan. Keep queries "
    "concise and varied."
)


def build_deep_search_prompt(event_group_prompt: str) -> str:
    """Return the event group prompt with deep search guidance prepended."""
    return f"{DEEP_SEARCH_INSTRUCTIONS}\n\nEvent group:\n{event_group_prompt}".strip()


@dataclass
class DenseNewsIndex:
    """Luxical + LEANN search index over NewsArticle entries."""

    articles: list["NewsArticle"]
    luxical_model_path: str | None = None
    luxical_model_id: str | None = None
    luxical_model_filename: str | None = None
    embedder: Embedder | None = None
    show_progress: bool = False
    backend_name: str = "hnsw"
    search_complexity: int = 64
    search_beam_width: int = 1
    search_prune_ratio: float = 0.0
    backend_searcher: LeannBackendSearcherInterface = field(init=False)
    passage_manager: PassageManager = field(init=False)
    article_by_id: dict[str, "NewsArticle"] = field(init=False)
    index_path: Path = field(init=False)
    temp_dir: tempfile.TemporaryDirectory | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Build the Luxical + LEANN index from the provided articles."""
        if self.embedder is None:
            self.embedder = self._load_embedder()
        self.article_by_id = {article.article_id: article for article in self.articles}
        self._build_index()

    def _load_embedder(self) -> Embedder:
        """Load a Luxical embedder from a local path or HuggingFace Hub."""
        if self.luxical_model_path is None:
            if self.luxical_model_id is None or self.luxical_model_filename is None:
                raise ValueError(
                    "luxical_model_path or luxical_model_id + luxical_model_filename is required."
                )
            logging.info(
                "Dense embedder download starting repo_id=%s filename=%s",
                self.luxical_model_id,
                self.luxical_model_filename,
            )
            started_at = time.perf_counter()
            self.luxical_model_path = hf_hub_download(
                repo_id=self.luxical_model_id,
                filename=self.luxical_model_filename,
            )
            elapsed = time.perf_counter() - started_at
            logging.info(
                "Dense embedder download completed repo_id=%s filename=%s seconds=%.2f",
                self.luxical_model_id,
                self.luxical_model_filename,
                elapsed,
            )
        return Embedder.load(self.luxical_model_path)

    def _build_index(self) -> None:
        """Create the dense vector index with Luxical embeddings."""
        logging.info("Dense index build starting articles=%s", len(self.articles))
        started_at = time.perf_counter()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_path = Path(self.temp_dir.name) / "news.leann"

        texts = [article.prompt_text() for article in self.articles]
        embeddings = self._embed_texts(texts, purpose="documents")
        embeddings_file = Path(self.temp_dir.name) / "news_embeddings.pkl"
        with open(embeddings_file, "wb") as handle:
            pickle.dump(([article.article_id for article in self.articles], embeddings), handle)

        builder = LeannBuilder(
            backend_name=self.backend_name,
            embedding_model="luxical",
            embedding_mode="luxical",
            is_recompute=False,
            is_compact=False,
            distance_metric="cosine",
        )
        for article in self.articles:
            builder.add_text(
                article.prompt_text(),
                metadata={
                    "id": article.article_id,
                    "article_id": article.article_id,
                    "title": article.title,
                    "desc": article.desc,
                    "published_at": article.published_at.isoformat(),
                    "url": article.url,
                    "domain": article.domain,
                    "outlet_name": article.outlet_name,
                },
            )
        builder.build_index_from_embeddings(str(self.index_path), str(embeddings_file))

        meta_path = Path(f"{self.index_path}.meta.json")
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        self.passage_manager = PassageManager(meta["passage_sources"], metadata_file_path=str(meta_path))
        backend_factory = BACKEND_REGISTRY[self.backend_name]
        self.backend_searcher = backend_factory.searcher(str(self.index_path))

        elapsed = time.perf_counter() - started_at
        logging.info(
            "Dense index build completed articles=%s seconds=%.2f",
            len(self.articles),
            elapsed,
        )

    def _embed_texts(self, texts: list[str], purpose: str) -> np.ndarray:
        """Return dense embeddings for the provided texts."""
        logging.info("Dense embeddings starting purpose=%s items=%s", purpose, len(texts))
        started_at = time.perf_counter()
        embeddings = self.embedder(texts, progress_bars=self.show_progress)
        elapsed = time.perf_counter() - started_at
        logging.info(
            "Dense embeddings completed purpose=%s items=%s seconds=%.2f",
            purpose,
            len(texts),
            elapsed,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def search(self, query: str, top_k: int) -> list["NewsArticle"]:
        """Return the top-k dense matches for a query."""
        query_embedding = self._embed_texts([query], purpose="query")
        logging.info("Dense search starting query=%s top_k=%s", query, top_k)
        started_at = time.perf_counter()
        results = self.backend_searcher.search(
            query_embedding,
            top_k,
            complexity=self.search_complexity,
            beam_width=self.search_beam_width,
            prune_ratio=self.search_prune_ratio,
            recompute_embeddings=False,
        )
        elapsed = time.perf_counter() - started_at
        logging.info(
            "Dense search completed query=%s top_k=%s seconds=%.2f n_results=%s",
            query,
            top_k,
            elapsed,
            len(results["labels"][0])
        )
        labels = results["labels"][0]
        return [self.article_by_id[str(label)] for label in labels]

    def article_for_id(self, article_id: str) -> "NewsArticle":
        """Return a NewsArticle by its id."""
        return self.article_by_id[article_id]


@dataclass
class DenseSearchTool:
    """Tool wrapper that tracks dense candidates for a DSPy agent."""

    index: DenseNewsIndex
    default_top_k: int = 25
    event_group_id: str | None = None
    candidate_ids: list[str] = field(default_factory=list)
    query_log: list[dict] = field(default_factory=list)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search the dense semantic index and return article summaries."""
        search_top_k = top_k if top_k is not None else self.default_top_k
        logged_at = dt.datetime.now(dt.timezone.utc).isoformat()
        logging.info(
            "Dense query event_group_id=%s query=%s top_k=%s",
            self.event_group_id,
            query,
            search_top_k,
        )
        results = self.index.search(query, search_top_k)
        summaries = []
        for article in results:
            if article.article_id not in self.candidate_ids:
                self.candidate_ids.append(article.article_id)
            summaries.append(
                {
                    "article_id": article.article_id,
                    "title": article.title,
                    "desc": article.desc,
                    "published_at": article.published_at.isoformat(),
                    "url": article.url,
                    "domain": article.domain,
                    "outlet_name": article.outlet_name,
                }
            )
        self.query_log.append(
            {
                "tool": "dense",
                "query": query,
                "top_k": search_top_k,
                "logged_at": logged_at,
                "results": summaries,
            }
        )
        return summaries

    def collected_articles(self) -> list["NewsArticle"]:
        """Return unique candidate articles collected during searches."""
        return [self.index.article_for_id(article_id) for article_id in self.candidate_ids]


class DenseDeepSearchAgent(dspy.Module):
    """DSPy module that uses dense search to identify relevant articles."""

    def __init__(self, model: str, search_tool: DenseSearchTool, max_iters: int) -> None:
        """Initialize the deep search agent with a model and dense tool."""
        super().__init__()
        self.model = model
        self.search_tool = search_tool
        self.max_iters = max_iters
        dspy.configure(lm=dspy.LM(model))
        self.module = dspy.ReAct(
            DeepSearchRelevantArticles,
            tools=[dspy.Tool(self.search_tool.search, name="dense_search")],
            max_iters=max_iters,
        )

    def forward(
        self,
        event_group_prompt: str,
        present_timeline: str,
        max_results: int,
    ) -> dspy.Prediction:
        """Run the agent and return the raw DSPy prediction."""
        dspy.configure(lm=dspy.LM(self.model))
        return self.module(
            event_group_prompt=build_deep_search_prompt(event_group_prompt),
            present_timeline=present_timeline,
            max_results=max_results,
        )

    def predict(
        self,
        event_group_prompt: str,
        present_timeline: str,
        max_results: int,
    ) -> dspy.Prediction:
        """Return the raw prediction for the deep search call."""
        return self(
            event_group_prompt=event_group_prompt,
            present_timeline=present_timeline,
            max_results=max_results,
        )

    def extract_relevant_ids(self, prediction: dspy.Prediction) -> list[str]:
        """Return relevant article ids from a DSPy prediction."""
        if prediction is None:
            raise ValueError("Deep search agent returned no result.")
        if prediction.relevant_article_ids is None:
            raise ValueError("Deep search agent returned no relevant_article_ids.")
        if not isinstance(prediction.relevant_article_ids, list):
            raise ValueError(
                "Deep search agent returned non-list relevant_article_ids "
                f"type={type(prediction.relevant_article_ids)} model={self.model}"
            )
        return prediction.relevant_article_ids

    def find_relevant(
        self,
        event_group_prompt: str,
        present_timeline: str,
        max_results: int,
    ) -> list[str]:
        """Return relevant article ids selected by the agent."""
        prediction = self.predict(
            event_group_prompt=event_group_prompt,
            present_timeline=present_timeline,
            max_results=max_results,
        )
        return self.extract_relevant_ids(prediction)


class DualSearchAgent(dspy.Module):
    """DSPy module that uses dense and BM25 search to identify relevant articles."""

    def __init__(
        self,
        model: str,
        bm25_tool: BM25SearchTool,
        dense_tool: DenseSearchTool,
        max_iters: int,
    ) -> None:
        """Initialize the deep search agent with both retrieval tools."""
        super().__init__()
        self.model = model
        self.bm25_tool = bm25_tool
        self.dense_tool = dense_tool
        self.max_iters = max_iters
        dspy.configure(lm=dspy.LM(model))
        self.module = dspy.ReAct(
            DeepSearchRelevantArticles,
            tools=[
                dspy.Tool(self.bm25_tool.search, name="bm25_search"),
                dspy.Tool(self.dense_tool.search, name="dense_search"),
            ],
            max_iters=max_iters,
        )

    def forward(
        self,
        event_group_prompt: str,
        present_timeline: str,
        max_results: int,
    ) -> dspy.Prediction:
        """Run the agent and return the raw DSPy prediction."""
        dspy.configure(lm=dspy.LM(self.model))
        return self.module(
            event_group_prompt=build_deep_search_prompt(event_group_prompt),
            present_timeline=present_timeline,
            max_results=max_results,
        )

    def predict(
        self,
        event_group_prompt: str,
        present_timeline: str,
        max_results: int,
    ) -> dspy.Prediction:
        """Return the raw prediction for the deep search call."""
        return self(
            event_group_prompt=event_group_prompt,
            present_timeline=present_timeline,
            max_results=max_results,
        )

    def extract_relevant_ids(self, prediction: dspy.Prediction) -> list[str]:
        """Return relevant article ids from a DSPy prediction."""
        if prediction is None:
            raise ValueError("Deep search agent returned no result.")
        if prediction.relevant_article_ids is None:
            raise ValueError("Deep search agent returned no relevant_article_ids.")
        if not isinstance(prediction.relevant_article_ids, list):
            raise ValueError(
                "Deep search agent returned non-list relevant_article_ids "
                f"type={type(prediction.relevant_article_ids)} model={self.model}"
            )
        return prediction.relevant_article_ids

    def find_relevant(
        self,
        event_group_prompt: str,
        present_timeline: str,
        max_results: int,
    ) -> list[str]:
        """Return relevant article ids selected by the agent."""
        prediction = self.predict(
            event_group_prompt=event_group_prompt,
            present_timeline=present_timeline,
            max_results=max_results,
        )
        return self.extract_relevant_ids(prediction)
