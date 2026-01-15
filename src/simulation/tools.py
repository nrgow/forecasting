import logging
import time
import httpx
from markdownify import markdownify
import wikipedia
import json
import sqlitedict


def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on timeline generation progress and decision-making.

    Use this tool after you recieve each new information to analyze results and plan next steps systematically.
    This creates a deliberate pause in the timeline generation workflow for quality decision-making.

    When to use:
    - After receiving search results: What key events did I find?
    - After reading a wikipedia page: What key dates and events did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing timeline gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information or timespans are still missing?
    3. Quality evaluation - Do I have sufficient breadth and depth for a good answer?
    4. Strategic decision - Should I add further detail to the timeline or provide my answer?

    Args:
        reflection: Your detailed reflection on timeline generation progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    logging.info(f"Calling the think tool with {reflection=}")
    return f"Reflection recorded: {reflection}"


class CachedWikipedia:
    """Cached access to Wikipedia search and page retrieval."""

    def __init__(self):
        """Initialize page and search caches."""
        self.page_cache = sqlitedict.SqliteDict("wikipedia_page_cache", autocommit=True)
        self.search_cache = sqlitedict.SqliteDict(
            "wikipedia_search_cache", autocommit=True
        )

    def _candidate_titles(self, page_title: str) -> list[str]:
        """Return ordered, unique title variants to try."""
        normalized = " ".join(page_title.split())
        variants = [
            normalized,
            normalized.replace("\u2013", "-").replace("\u2014", "-"),
            normalized.replace("-", "\u2013"),
            normalized.replace("-", "\u2014"),
        ]
        seen = set()
        ordered = []
        for value in variants:
            if value not in seen:
                ordered.append(value)
                seen.add(value)
        return ordered

    def _select_search_title(self, term: str, results: list[str]) -> str | None:
        """Select the best title from search results."""
        if not results:
            return None
        term_key = term.casefold()
        for title in results:
            if title.casefold() == term_key:
                return title
        return results[0]

    def get_wikipedia_page(self, page_title: str) -> str:
        """Return a markdown version of a Wikipedia page by title.

        Args:
            page_title: Wikipedia page title to load.

        Returns:
            Page content converted to markdown.
        """
        if (rval := self.page_cache.get(page_title)) is not None:
            return rval
        logging.info("called get_wikipedia_page with parameters %r", page_title)
        started_at = time.perf_counter()
        last_error: Exception | None = None

        for candidate in self._candidate_titles(page_title):
            try:
                page = wikipedia.page(candidate, auto_suggest=False)
                rval = markdownify(page.html())
                self.page_cache[page_title] = rval
                elapsed = time.perf_counter() - started_at
                logging.info(
                    "get_wikipedia_page completed title=%s seconds=%.2f",
                    candidate,
                    elapsed,
                )
                return rval
            except wikipedia.exceptions.DisambiguationError as exc:
                logging.info(
                    "get_wikipedia_page disambiguation title=%s options=%s",
                    candidate,
                    len(exc.options),
                )
                last_error = exc
            except wikipedia.exceptions.WikipediaException as exc:
                logging.info(
                    "get_wikipedia_page failed title=%s error=%s", candidate, exc
                )
                last_error = exc

        search_term = " ".join(page_title.split())
        logging.info("get_wikipedia_page search fallback term=%s", search_term)
        try:
            results = wikipedia.search(search_term, results=10)
        except wikipedia.exceptions.WikipediaException as exc:
            last_error = exc
            return str(last_error)

        best_title = self._select_search_title(search_term, results)
        if best_title is None:
            if last_error is not None:
                return str(last_error)
            return f"No Wikipedia page found for {page_title!r}."

        try:
            page = wikipedia.page(best_title, auto_suggest=False)
        except wikipedia.exceptions.WikipediaException as exc:
            last_error = exc
            return str(last_error)

        rval = markdownify(page.html())
        self.page_cache[page_title] = rval
        elapsed = time.perf_counter() - started_at
        logging.info(
            "get_wikipedia_page completed title=%s seconds=%.2f", best_title, elapsed
        )
        return rval

    def search_wikipedia_pages(self, term: str, n_results: int = 10) -> list[str]:
        """Search Wikipedia for page titles matching a query.

        Args:
            term: Search phrase.
            n_results: The number of search results to deliver.

        Returns:
            List of potentially relevant Wikipedia page titles.
        """
        logging.info(
            f"called search_wikipedia_pages with parameters {term=} {n_results=}"
        )
        if (rval := self.search_cache.get(term)) is not None:
            return json.loads(rval)
        rval = wikipedia.search(term, results=n_results)
        self.search_cache[term] = json.dumps(rval)
        return rval


def tool_search(tool_description: str):
    """query available tools"""
    logging.info(f"called tool_search with parameters {repr(tool_description)}")
    return "ERROR: the toolsearch tool is currently experiencing an outage"


def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Webpage content as markdown
    """
    logging.info("fetch_webpage_content(%s)", url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        logging.exception(e)
        logging.info("fetch_webpage_content: Error")
        return f"Error fetching content from {url}: {str(e)}"
