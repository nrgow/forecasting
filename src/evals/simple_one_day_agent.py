import logging
import os
import time
from typing import Any

import httpx
from openai import OpenAI
import json

from ..simulation.tools import CachedWikipedia


class SimpleOneDayAgent:
    """Lightweight tool-calling agent for one-day summaries (Wikipedia + Exa)."""

    def __init__(
        self,
        model: str = "x-ai/grok-2-latest",
        temperature: float = 0.2,
        max_tokens: int = 1200,
        max_iters: int = 6,
        exa_api_url: str = "https://api.exa.ai/search",
        add_reasoning_content: bool = True,
    ) -> None:
        """Initialize the agent with model settings and tool helpers."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iters = max_iters
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.wiki = CachedWikipedia()
        self.exa_api_url = exa_api_url
        self.exa_api_key = os.environ.get("EXA_API_KEY")
        self.add_reasoning_content = add_reasoning_content

    def _tools(self) -> list[dict[str, Any]]:
        """Return OpenAI tool specs."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "wikipedia_search",
                    "description": "Search Wikipedia for relevant page titles.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "wikipedia_page",
                    "description": "Fetch a Wikipedia page and return markdown.",
                    "parameters": {
                        "type": "object",
                        "properties": {"page": {"type": "string"}},
                        "required": ["page"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "exa_search",
                    "description": "Search Exa for recent web results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "num_results": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Return the final one-day summary in markdown.",
                    "parameters": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            },
        ]

    def _handle_wikipedia_search(self, query: str) -> str:
        """Return JSON list of wiki titles."""
        logging.info("simple-one-day wikipedia_search %s", query)
        started_at = time.perf_counter()
        titles = self.wiki.search_wikipedia_pages(query, 5)
        logging.info(
            "simple-one-day wikipedia_search done seconds=%.2f results=%s",
            time.perf_counter() - started_at,
            len(titles),
        )
        return json.dumps(titles, ensure_ascii=True)

    def _handle_wikipedia_page(self, page: str) -> str:
        """Return markdown page content."""
        logging.info("simple-one-day wikipedia_page %s", page)
        started_at = time.perf_counter()
        content = self.wiki.get_wikipedia_page(page)
        logging.info(
            "simple-one-day wikipedia_page done seconds=%.2f chars=%s",
            time.perf_counter() - started_at,
            len(content),
        )
        return content

    def _handle_exa_search(self, query: str, num_results: int = 5) -> str:
        """Query Exa and return JSON lines of title/url/snippet."""
        logging.info("simple-one-day exa_search %s", query)
        if not self.exa_api_key:
            return "EXA_API_KEY not set"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.exa_api_key,
        }
        payload = {"query": query, "numResults": num_results}
        started_at = time.perf_counter()
        try:
            response = httpx.post(
                self.exa_api_url, headers=headers, json=payload, timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            lines = [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("snippet"),
                }
                for item in results
            ]
            logging.info(
                "simple-one-day exa_search done seconds=%.2f results=%s",
                time.perf_counter() - started_at,
                len(lines),
            )
            return json.dumps(lines, ensure_ascii=True)
        except Exception as exc:
            logging.warning("simple-one-day exa_search error=%s", exc)
            return f"exa_search_error: {exc}"

    def run(self, topic: str, target_date: str) -> str:
        """Run the agent to produce a one-day summary."""
        system_prompt = (
            "You are a focused research agent. Build a concise one-day summary.\n"
            f"Date: {target_date}\nTopic: {topic}\n"
            "- Use wikipedia_search / wikipedia_page for background and names.\n"
            "- Use exa_search for the freshest reports.\n"
            "- Output markdown: heading '## {date}' then bullet points with concrete facts, numbers, locations, actors.\n"
            "- Keep under 180 words. If same-day info is missing, say 'Insufficient same-day information'."
        )
        messages = [{"role": "system", "content": system_prompt}]
        for step in range(1, self.max_iters + 1):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._tools(),
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            message = completion.choices[0].message
            if message.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
                if self.add_reasoning_content:
                    assistant_msg["reasoning_content"] = "Selecting tools to gather same-day facts."
                messages.append(assistant_msg)
                for call in message.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments)
                    if name == "wikipedia_search":
                        content = self._handle_wikipedia_search(args["query"])
                    elif name == "wikipedia_page":
                        content = self._handle_wikipedia_page(args["page"])
                    elif name == "exa_search":
                        content = self._handle_exa_search(
                            args["query"], args.get("num_results", 5)
                        )
                    elif name == "final_answer":
                        return args["answer"]
                    else:
                        content = f"unknown tool {name}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": content,
                        }
                    )
                continue
            if message.content:
                # If the model emits a direct answer without tool calls, accept it.
                return message.content
        raise RuntimeError("simple one-day agent failed to produce an answer")


def generate_one_day_simple_agent(
    topic: str,
    target_date: str,
    model: str = "x-ai/grok-2-latest",
    temperature: float = 0.2,
    max_tokens: int = 1200,
    max_iters: int = 6,
    add_reasoning_content: bool = True,
) -> dict[str, Any]:
    """Run the simple agent and return timeline and metadata."""
    agent = SimpleOneDayAgent(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_iters=max_iters,
        add_reasoning_content=add_reasoning_content,
    )
    timeline = agent.run(topic, target_date)
    return {
        "timeline": timeline,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_iters": max_iters,
        "target_date": target_date,
    }
