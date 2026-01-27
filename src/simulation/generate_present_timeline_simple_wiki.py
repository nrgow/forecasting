import json
import datetime as dt
import logging
import os
import time

from openai import OpenAI

from ..mlflow_tracing import log_inference_call
from . import tools


def build_simple_wiki_timeline_prompt(
    topic_pertaining_to: str, start_date: str, current_date: str
) -> str:
    """Return the prompt used for the simple wiki present timeline agent."""
    return (
        "Construct a chronological, information-dense timeline that begins on "
        f"{start_date} and runs through {current_date}.\n"        
        "Focus on factual, verifiable events and provide background context relevant to the topic.\n"
        "Include related parallel events only if they plausibly influence the main topic.\n"
        "You must cover recent events! Use the tools available to search for potential content. If you have trouble finding content, expand your search.\n"        
        "Do not include analysis, predictions, or opinion.\n"
        "Do not include links.\n"
        "Format the timeline in markdown with date headers as subsections (e.g., '## YYYY-MM-DD').\n"
        "When you are finished, please call the `final_answer` with the finished timeline!\n"        
        "Topic:\n"
        "{topic_pertaining_to}"
    )


class SimpleWikiWorkspace:
    """Track planning workspace for the simple wiki timeline agent."""

    def __init__(self, topic: str) -> None:
        """Initialize the workspace with the root topic."""
        self.topic = topic
        self.topics: list[str] = []
        self.entities: list[str] = []
        self.fetched_pages: list[str] = []
        self.gaps: list[str] = []
        self.notes: list[str] = []

    def apply_update(self, update: dict) -> None:
        """Apply a structured update to the workspace."""
        for key in ("topics", "entities", "fetched_pages", "gaps", "notes"):
            if key in update:
                getattr(self, key).extend(update[key])

    def snapshot(self) -> str:
        """Return a JSON snapshot of the workspace for prompting."""
        return json.dumps(
            {
                "topic": self.topic,
                "topics": self.topics,
                "entities": self.entities,
                "fetched_pages": self.fetched_pages,
                "gaps": self.gaps,
                "notes": self.notes,
            },
            ensure_ascii=True,
        )


class SimpleWikiTimelineAgent:
    """Tool-calling agent that builds present timelines from Wikipedia sources."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        max_iters: int,
    ) -> None:
        """Initialize the agent with OpenRouter-backed OpenAI settings."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iters = max_iters
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.wiki = tools.CachedWikipedia()

    def _tool_definitions(self) -> list[dict]:
        """Return tool schemas for the OpenAI tool-calling loop."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "wikipedia_search",
                    "description": "Search Wikipedia for relevant page titles.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "wikipedia_page",
                    "description": "Fetch a Wikipedia page and return its markdown content.",
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
                    "name": "think",
                    "description": "Record a brief reflection about findings and next steps.",
                    "parameters": {
                        "type": "object",
                        "properties": {"thoughts": {"type": "string"}},
                        "required": ["thoughts"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "workspace_update",
                    "description": (
                        "Update the planning workspace with new topics, entities, fetched pages, or gaps."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topics": {"type": "array", "items": {"type": "string"}},
                            "entities": {"type": "array", "items": {"type": "string"}},
                            "fetched_pages": {"type": "array", "items": {"type": "string"}},
                            "gaps": {"type": "array", "items": {"type": "string"}},
                            "notes": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": (
                        "Return the final timeline in markdown under the `answer` field."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            },
        ]

    def _think_only_tool_definitions(self) -> list[dict]:
        """Return tool schema restricted to the think tool."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "think",
                    "description": "Record a brief reflection about findings and next steps.",
                    "parameters": {
                        "type": "object",
                        "properties": {"thoughts": {"type": "string"}},
                        "required": ["thoughts"],
                    },
                },
            }
        ]

    def _handle_wikipedia_search(self, query: str) -> str:
        """Run Wikipedia search and return results as JSON."""
        logging.info("Simple wiki tool wikipedia_search starting query=%s", query)
        started_at = time.perf_counter()
        results = self.wiki.search_wikipedia_pages(query, 5)
        elapsed = time.perf_counter() - started_at
        logging.info(
            "Simple wiki tool wikipedia_search completed seconds=%.2f results=%s",
            elapsed,
            len(results),
        )
        return json.dumps(results, ensure_ascii=True)

    def _handle_wikipedia_page(self, page: str) -> str:
        """Fetch a Wikipedia page and return markdown content."""
        logging.info("Simple wiki tool wikipedia_page starting page=%s", page)
        started_at = time.perf_counter()
        content = self.wiki.get_wikipedia_page(page)
        elapsed = time.perf_counter() - started_at
        logging.info(
            "Simple wiki tool wikipedia_page completed seconds=%.2f chars=%s",
            elapsed,
            len(content),
        )
        return content

    def _handle_think(self, thoughts: str) -> str:
        """Record tool-based reflection for tracing."""
        logging.info("Simple wiki tool think called thoughts=%s", thoughts)
        return "Reflection recorded."

    def _handle_workspace_update(self, update: dict) -> str:
        """Apply an update to the workspace and return a summary."""
        logging.info("Simple wiki tool workspace_update starting")
        started_at = time.perf_counter()
        self.workspace.apply_update(update)
        elapsed = time.perf_counter() - started_at
        logging.info(
            "Simple wiki tool workspace_update completed seconds=%.2f topics=%s entities=%s fetched_pages=%s gaps=%s notes=%s",
            elapsed,
            len(self.workspace.topics),
            len(self.workspace.entities),
            len(self.workspace.fetched_pages),
            len(self.workspace.gaps),
            len(self.workspace.notes),
        )
        return "Workspace updated."

    def _append_tool_message(
        self,
        messages: list[dict],
        tool_call_id: str,
        content: str,
    ) -> None:
        """Append a tool response message to the conversation."""
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def _execute_tool_calls(
        self,
        tool_calls: list,
        messages: list[dict],
    ) -> str | None:
        """Execute tool calls and return a final answer if requested."""
        for call in tool_calls:
            tool_name = call.function.name
            arguments = json.loads(call.function.arguments)
            logging.info("Simple wiki tool call received name=%s", tool_name)
            if tool_name == "wikipedia_search":
                content = self._handle_wikipedia_search(arguments["query"])
                self._append_tool_message(messages, call.id, content)
            elif tool_name == "wikipedia_page":
                content = self._handle_wikipedia_page(arguments["page"])
                self._append_tool_message(messages, call.id, content)
            elif tool_name == "think":
                content = self._handle_think(arguments["thoughts"])
                self._append_tool_message(messages, call.id, content)
            elif tool_name == "workspace_update":
                content = self._handle_workspace_update(arguments)
                self._append_tool_message(messages, call.id, content)
            elif tool_name == "final_answer":
                return arguments["answer"]
            else:
                raise ValueError(f"Unknown tool call: {tool_name}")
        return None

    def run(
        self,
        topic_pertaining_to: str,
        current_date: str,
        event_group_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        """Run the tool-calling loop and return the generated timeline."""
        start_date = (dt.date.fromisoformat(current_date) - dt.timedelta(days=730)).isoformat()
        prompt = build_simple_wiki_timeline_prompt(
            topic_pertaining_to, start_date, current_date
        )
        system_message = (
            "You are a research agent assembling a factual, date-ordered timeline. "
            "Use wikipedia_search and wikipedia_page to gather evidence. "
            "Maintain the planning workspace using workspace_update with topics, entities, fetched pages, and gaps. "
            "After any wikipedia_page tool calls, call think with a brief gap analysis before doing more retrieval. "
            "When finished, call final_answer with the markdown timeline in the `answer` field only."
        )
        messages: list[dict] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        token_budget = self.max_tokens
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        started_at = time.perf_counter()
        final_answer = None
        force_think = False
        max_retries = 3
        retry_delay_seconds = 1.0
        self.workspace = SimpleWikiWorkspace(topic_pertaining_to)

        logging.info("Simple wiki timeline agent starting topic=%s", topic_pertaining_to)
        for step in range(1, self.max_iters + 1):
            remaining_tokens = token_budget - total_usage["total_tokens"]
            percent_used = (total_usage["total_tokens"] / token_budget) * 100
            tools_schema = (
                self._think_only_tool_definitions()
                if force_think
                else self._tool_definitions()
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Token budget reminder: "
                        f"{remaining_tokens} total tokens remain in the configured budget. "
                        f"Budget used: {percent_used:.1f}%. "
                        f"Steps completed: {step - 1} of {self.max_iters}.\n"
                        "Be concise and prioritize the most important and recent events. "
                        "If finished, call final_answer with JSON: {\"answer\": \"...\"}."
                    ),
                }
            )
            #messages.append(
            #    {
            #        "role": "system",
            #        "content": (
            #            "Workspace snapshot (JSON): "
            #            f"{self.workspace.snapshot()}\n"
            #            "Use workspace_update to add topics, entities, fetched pages, or gaps."
            #        ),
            #    }
            #)
            if force_think:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You must call think now with a short gap analysis "
                            "(e.g., missing last 90 days, missing subtopic)."
                        ),
                    }
                )
            logging.info("Simple wiki timeline LLM call starting step=%s", step)
            completion = None
            last_error: Exception | None = None
            for attempt in range(1, max_retries + 1):
                step_started_at = time.perf_counter()
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools_schema,
                        tool_choice="required",
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    if completion is None or completion.choices is None:
                        payload = None
                        if completion is not None and hasattr(completion, "model_dump"):
                            payload = completion.model_dump()
                        logging.error(
                            "Simple wiki timeline missing choices step=%s attempt=%s payload=%s",
                            step,
                            attempt,
                            payload,
                        )
                        raise RuntimeError("OpenRouter response missing choices.")
                    break
                except Exception as exc:
                    last_error = exc
                    step_elapsed = time.perf_counter() - step_started_at
                    logging.warning(
                        "Simple wiki timeline LLM call failed step=%s attempt=%s seconds=%.2f error=%s",
                        step,
                        attempt,
                        step_elapsed,
                        exc,
                    )
                    if attempt < max_retries:
                        time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))
            if completion is None or completion.choices is None:
                raise RuntimeError("OpenRouter response missing choices.") from last_error
            step_elapsed = time.perf_counter() - step_started_at
            message = completion.choices[0].message
            usage = completion.usage
            if usage is not None:
                total_usage["prompt_tokens"] += usage.prompt_tokens
                total_usage["completion_tokens"] += usage.completion_tokens
                total_usage["total_tokens"] += usage.total_tokens
            logging.info(
                "Simple wiki timeline LLM call completed step=%s seconds=%.2f prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                step,
                step_elapsed,
                usage.prompt_tokens if usage is not None else None,
                usage.completion_tokens if usage is not None else None,
                usage.total_tokens if usage is not None else None,
            )
            log_inference_call(
                name="present_timeline.simple_wiki",
                model=self.model,
                inputs={
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "step": step,
                },
                outputs={
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        }
                        for call in (message.tool_calls or [])
                    ],
                },
                metadata={
                    "pipeline_run_id": run_id,
                    "event_group_id": event_group_id,
                    "stage": "react_loop",
                    "step": step,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage is not None else None,
                        "completion_tokens": usage.completion_tokens
                        if usage is not None
                        else None,
                        "total_tokens": usage.total_tokens if usage is not None else None,
                    },
                },
                duration_seconds=step_elapsed,
            )
            if message.tool_calls:
                messages.append(
                    {
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
                )
                final_answer = self._execute_tool_calls(message.tool_calls, messages)
                if final_answer is not None:
                    break
                saw_think = any(
                    call.function.name == "think" for call in message.tool_calls
                )
                saw_page = any(
                    call.function.name == "wikipedia_page"
                    for call in message.tool_calls
                )
                if force_think and not saw_think:
                    logging.warning(
                        "Simple wiki timeline expected think tool call but did not receive it step=%s",
                        step,
                    )
                force_think = saw_page and not saw_think
                continue
            if message.content:
                logging.warning(
                    "Simple wiki timeline assistant returned content without final_answer tool step=%s",
                    step,
                )

        if final_answer is None:
            raise RuntimeError("Simple wiki timeline agent did not produce an answer.")

        total_elapsed = time.perf_counter() - started_at
        logging.info(
            "Simple wiki timeline agent completed seconds=%.2f total_prompt_tokens=%s total_completion_tokens=%s total_tokens=%s",
            total_elapsed,
            total_usage["prompt_tokens"],
            total_usage["completion_tokens"],
            total_usage["total_tokens"],
        )
        return {
            "timeline": final_answer,
            "prompt": prompt,
            "model": self.model,
            "current_date": current_date,
            "usage": total_usage,
            "elapsed_seconds": total_elapsed,
        }


def generate_present_timeline_simple_wiki(
    topic_pertaining_to: str,
    current_date: str,
    model: str = "x-ai/grok-4-fast",
    temperature: float = 0.2,
    max_tokens: int = 32000,
    max_iters: int = 20,
    event_group_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Generate a present timeline with a simple Wikipedia tool loop."""
    agent = SimpleWikiTimelineAgent(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_iters=max_iters,
    )
    return agent.run(
        topic_pertaining_to=topic_pertaining_to,
        current_date=current_date,
        event_group_id=event_group_id,
        run_id=run_id,
    )
