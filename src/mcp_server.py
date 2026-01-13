import argparse
import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastmcp import FastMCP

from src.simulation.bm25_retriever import BM25NewsIndex, BM25SearchTool
from src.simulation.dense_retriever import DenseNewsIndex, DenseSearchTool
from src.simulation.simulation_pipeline import iter_news_articles, iter_news_paths
from src.simulation.tools import CachedWikipedia


@dataclass
class RetrievalState:
    """Preloaded retrieval state for MCP tools."""

    news_base_path: Path
    news_since_hours: int | None
    luxical_model_path: str | None
    luxical_model_id: str | None
    luxical_model_filename: str | None
    dense_search_complexity: int
    dense_search_beam_width: int
    dense_search_prune_ratio: float
    bm25_default_top_k: int
    dense_default_top_k: int
    show_progress: bool = False
    bm25_tool: BM25SearchTool = field(init=False)
    dense_tool: DenseSearchTool = field(init=False)

    def __post_init__(self) -> None:
        """Load articles and build BM25 + dense indexes."""
        logging.info("MCP retrieval state initialization starting")
        started_at = time.perf_counter()
        articles = self._load_articles()

        bm25_started_at = time.perf_counter()
        bm25_index = BM25NewsIndex(articles, show_progress=self.show_progress)
        bm25_elapsed = time.perf_counter() - bm25_started_at
        logging.info(
            "MCP BM25 index ready articles=%s seconds=%.2f",
            len(articles),
            bm25_elapsed,
        )
        self.bm25_tool = BM25SearchTool(bm25_index, default_top_k=self.bm25_default_top_k)

        dense_started_at = time.perf_counter()
        dense_index = DenseNewsIndex(
            articles,
            luxical_model_path=self.luxical_model_path,
            luxical_model_id=self.luxical_model_id,
            luxical_model_filename=self.luxical_model_filename,
            show_progress=self.show_progress,
            search_complexity=self.dense_search_complexity,
            search_beam_width=self.dense_search_beam_width,
            search_prune_ratio=self.dense_search_prune_ratio,
        )
        dense_elapsed = time.perf_counter() - dense_started_at
        logging.info(
            "MCP dense index ready articles=%s seconds=%.2f",
            len(articles),
            dense_elapsed,
        )
        self.dense_tool = DenseSearchTool(dense_index, default_top_k=self.dense_default_top_k)

        elapsed = time.perf_counter() - started_at
        logging.info("MCP retrieval state initialized seconds=%.2f", elapsed)

    def _load_articles(self) -> list:
        """Load news articles from the base path and optional time window."""
        since = None
        if self.news_since_hours is not None:
            since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=self.news_since_hours)
        logging.info(
            "MCP article load starting base_path=%s since=%s",
            self.news_base_path,
            since.isoformat() if since is not None else None,
        )
        started_at = time.perf_counter()
        news_paths = iter_news_paths(self.news_base_path, since)
        articles = list(iter_news_articles(news_paths, since))
        elapsed = time.perf_counter() - started_at
        logging.info(
            "MCP article load completed paths=%s articles=%s seconds=%.2f",
            len(news_paths),
            len(articles),
            elapsed,
        )
        return articles


def build_mcp(state: RetrievalState, enabled_tools: set[str]) -> FastMCP:
    """Create the FastMCP server with retrieval tools."""
    mcp = FastMCP("newstalk")
    wiki = CachedWikipedia()

    if "wikipedia_search" in enabled_tools:
        @mcp.tool()
        def wikipedia_search(term: str, n_results: int = 10) -> list[str]:
            """Search Wikipedia for page titles."""
            started_at = time.perf_counter()
            logging.info("MCP wikipedia_search starting term=%s n_results=%s", term, n_results)
            results = wiki.search_wikipedia_pages(term, n_results)
            elapsed = time.perf_counter() - started_at
            logging.info(
                "MCP wikipedia_search completed term=%s results=%s seconds=%.2f",
                term,
                len(results),
                elapsed,
            )
            return results

    if "wikipedia_page" in enabled_tools:
        @mcp.tool()
        def wikipedia_page(title: str) -> str:
            """Return the markdown for a Wikipedia page."""
            started_at = time.perf_counter()
            logging.info("MCP wikipedia_page starting title=%s", title)
            page = wiki.get_wikipedia_page(title)
            elapsed = time.perf_counter() - started_at
            logging.info("MCP wikipedia_page completed title=%s seconds=%.2f", title, elapsed)
            return page

    if "bm25_search" in enabled_tools:
        @mcp.tool()
        def bm25_search(query: str, top_k: int | None = None) -> list[dict]:
            """Search the BM25 index and return article summaries."""
            started_at = time.perf_counter()
            logging.info("MCP bm25_search starting query=%s top_k=%s", query, top_k)
            results = state.bm25_tool.search(query, top_k)
            elapsed = time.perf_counter() - started_at
            logging.info(
                "MCP bm25_search completed query=%s results=%s seconds=%.2f",
                query,
                len(results),
                elapsed,
            )
            return results

    if "dense_search" in enabled_tools:
        @mcp.tool()
        def dense_search(query: str, top_k: int | None = None) -> list[dict]:
            """Search the dense index and return article summaries."""
            started_at = time.perf_counter()
            logging.info("MCP dense_search starting query=%s top_k=%s", query, top_k)
            results = state.dense_tool.search(query, top_k)
            elapsed = time.perf_counter() - started_at
            logging.info(
                "MCP dense_search completed query=%s results=%s seconds=%.2f",
                query,
                len(results),
                elapsed,
            )
            return results

    return mcp


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MCP server."""
    parser = argparse.ArgumentParser(description="Newstalk MCP server for retrieval tools.")
    parser.add_argument(
        "--tools",
        type=str,
        default="wikipedia_search,wikipedia_page,bm25_search,dense_search",
        help="Comma-separated list of MCP tools to expose.",
    )
    parser.add_argument(
        "--news-base-path",
        type=Path,
        default=Path("/mnt/ssd") / "newstalk-data" / "gdelt-gal",
        help="Path to the GDELT news data directory.",
    )
    parser.add_argument(
        "--news-since-hours",
        type=int,
        default=72,
        help="Limit articles to the last N hours; set to 0 to load all.",
    )
    parser.add_argument(
        "--luxical-model-path",
        type=str,
        default=None,
        help="Local path to the Luxical embedding model.",
    )
    parser.add_argument(
        "--luxical-model-id",
        type=str,
        default="DatologyAI/luxical-one",
        help="HuggingFace repo id for the Luxical model.",
    )
    parser.add_argument(
        "--luxical-model-filename",
        type=str,
        default="luxical_one_rc4.npz",
        help="HuggingFace filename for the Luxical model.",
    )
    parser.add_argument(
        "--dense-search-complexity",
        type=int,
        default=64,
        help="Dense retrieval search complexity.",
    )
    parser.add_argument(
        "--dense-search-beam-width",
        type=int,
        default=1,
        help="Dense retrieval beam width.",
    )
    parser.add_argument(
        "--dense-search-prune-ratio",
        type=float,
        default=0.0,
        help="Dense retrieval prune ratio.",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=25,
        help="Default BM25 top_k for searches.",
    )
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=25,
        help="Default dense top_k for searches.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Enable progress indicators during index construction.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level for the server.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the MCP server."""
    args = parse_args()
    log_level = logging._nameToLevel[args.log_level.upper()]
    logging.basicConfig(level=log_level)
    enabled_tools = {item.strip() for item in args.tools.split(",") if item.strip()}
    if not enabled_tools:
        raise ValueError("At least one MCP tool must be enabled via --tools.")
    news_since_hours = args.news_since_hours if args.news_since_hours != 0 else None
    state = RetrievalState(
        news_base_path=args.news_base_path,
        news_since_hours=news_since_hours,
        luxical_model_path=args.luxical_model_path,
        luxical_model_id=args.luxical_model_id,
        luxical_model_filename=args.luxical_model_filename,
        dense_search_complexity=args.dense_search_complexity,
        dense_search_beam_width=args.dense_search_beam_width,
        dense_search_prune_ratio=args.dense_search_prune_ratio,
        bm25_default_top_k=args.bm25_top_k,
        dense_default_top_k=args.dense_top_k,
        show_progress=args.show_progress,
    )
    mcp = build_mcp(state, enabled_tools)
    mcp.run()


if __name__ == "__main__":
    main()
