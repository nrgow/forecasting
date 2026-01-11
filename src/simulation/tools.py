import logging
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
    def __init__(self):
        self.page_cache = sqlitedict.SqliteDict("wikipedia_page_cache", autocommit=True)
        self.search_cache = sqlitedict.SqliteDict("wikipedia_search_cache", autocommit=True)        

    def get_wikipedia_page(self, page_title: str) -> str:
        """Return a markdown version of a Wikipedia page by title.

        Args:
            page_title: Wikipedia page title to load.

        Returns:
            Page content converted to markdown.
        """
        if (rval := self.page_cache.get(page_title)) is not None:
            return rval
        logging.info(f"called get_wikipedia_page with parameters {repr(page_title)}")    
        try:
            page = wikipedia.page(page_title)
        except wikipedia.exceptions.WikipediaException as e:
            return str(e)
        rval = markdownify(page.html())
        self.page_cache[page_title] = rval
        return rval

    def search_wikipedia_pages(self, term: str, n_results: int = 10) -> list[str]:
        """Search Wikipedia for page titles matching a query.

        Args:
            term: Search phrase.
            n_results: The number of search results to deliver.

        Returns:
            List of potentially relevant Wikipedia page titles.
        """
        logging.info(f"called search_wikipedia_pages with parameters {term=} {n_results=}")
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