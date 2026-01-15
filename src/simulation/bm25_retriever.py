from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import logging
import time
from typing import TYPE_CHECKING

import bm25s
import dspy

from .news_relevance_dspy import DeepSearchRelevantArticles
from ..mlflow_tracing import configure_dspy_autolog

if TYPE_CHECKING:
    from .simulation_pipeline import NewsArticle


@dataclass
class BM25NewsIndex:
    """BM25-backed search index over NewsArticle entries."""

    articles: list["NewsArticle"]
    show_progress: bool = False
    retriever: bm25s.BM25 = field(init=False)
    article_by_id: dict[str, "NewsArticle"] = field(init=False)

    def __post_init__(self) -> None:
        """Build the BM25 index from the provided articles."""
        logging.info("BM25 index build starting articles=%s", len(self.articles))
        started_at = time.perf_counter()
        self.article_by_id = {article.article_id: article for article in self.articles}
        corpus = [article.prompt_text() for article in self.articles]
        tokenized = bm25s.tokenize(
            corpus, return_ids=True, show_progress=self.show_progress
        )
        self.retriever = bm25s.BM25(corpus=self.articles)
        self.retriever.index(tokenized, show_progress=self.show_progress)
        elapsed = time.perf_counter() - started_at
        logging.info(
            "BM25 index build completed articles=%s seconds=%.2f",
            len(self.articles),
            elapsed,
        )

    def search(self, query: str, top_k: int) -> list["NewsArticle"]:
        """Return the top-k BM25 matches for a query."""
        tokenized_query = bm25s.tokenize(query, return_ids=True, show_progress=False)
        results = self.retriever.retrieve(
            tokenized_query,
            corpus=self.articles,
            k=top_k,
            show_progress=False,
            return_as="tuple",
        )
        return results.documents[0].tolist()

    def article_for_id(self, article_id: str) -> "NewsArticle":
        """Return a NewsArticle by its id."""
        return self.article_by_id[article_id]


@dataclass
class BM25SearchTool:
    """Tool wrapper that tracks BM25 candidates for a DSPy agent."""

    index: BM25NewsIndex
    default_top_k: int = 25
    event_group_id: str | None = None
    candidate_ids: list[str] = field(default_factory=list)
    query_log: list[dict] = field(default_factory=list)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search the lexical BM25 index and return article summaries."""
        search_top_k = top_k if top_k is not None else self.default_top_k
        logged_at = dt.datetime.now(dt.timezone.utc).isoformat()
        logging.info(
            "BM25 query event_group_id=%s query=%s top_k=%s",
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
                "tool": "bm25",
                "query": query,
                "top_k": search_top_k,
                "logged_at": logged_at,
                "results": summaries,
            }
        )
        return summaries

    def collected_articles(self) -> list["NewsArticle"]:
        """Return unique candidate articles collected during searches."""
        return [
            self.index.article_for_id(article_id) for article_id in self.candidate_ids
        ]


class DeepSearchAgent(dspy.Module):
    """DSPy module that uses BM25 search to identify relevant articles."""

    def __init__(self, model: str, search_tool: BM25SearchTool, max_iters: int) -> None:
        """Initialize the deep search agent with a model and BM25 tool."""
        super().__init__()
        self.model = model
        self.search_tool = search_tool
        self.max_iters = max_iters
        configure_dspy_autolog()
        dspy.configure(lm=dspy.LM(model))
        self.module = dspy.ReAct(
            DeepSearchRelevantArticles,
            tools=[dspy.Tool(self.search_tool.search, name="bm25_search")],
            max_iters=max_iters,
        )

    def forward(
        self,
        event_group_prompt: str,
        present_timeline: str,
        max_results: int,
    ) -> dspy.Prediction:
        """Run the agent and return the raw DSPy prediction."""
        configure_dspy_autolog()
        dspy.configure(lm=dspy.LM(self.model))
        return self.module(
            event_group_prompt=event_group_prompt,
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
