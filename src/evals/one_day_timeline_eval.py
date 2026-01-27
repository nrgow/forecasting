import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..mlflow_tracing import log_inference_call
from ..simulation.generate_present_timeline import generate_present_timeline
from ..simulation.generate_present_timeline_perplexity import (
    generate_present_timeline_perplexity,
)
from ..simulation.generate_present_timeline_simple_wiki import (
    generate_present_timeline_simple_wiki,
)
from .simple_one_day_agent import generate_one_day_simple_agent


def _usage_to_dict(usage: Any) -> dict[str, int | None]:
    """Convert OpenAI usage objects into a plain dict."""
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


@dataclass
class OneDayCase:
    """Single evaluation case containing a topic and target date (reference_path ignored)."""

    case_id: str
    topic: str
    target_date: str
    reference_path: Path | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OneDayCase":
        """Construct a case from a JSON payload."""
        reference_path = None
        if "reference_path" in payload and payload["reference_path"]:
            reference_path = Path(payload["reference_path"])
        return cls(
            case_id=payload["case_id"],
            topic=payload["topic"],
            target_date=payload["target_date"],
            reference_path=reference_path,
        )


@dataclass
class CandidateConfig:
    """Configuration for a single generator variant."""

    candidate_id: str
    generator: str
    model: str
    description: str


@dataclass
class CandidateRun:
    """Container for generator outputs and timing."""

    timeline: str
    summary: str
    generator_metadata: dict[str, Any]
    summary_usage: dict[str, int | None]
    elapsed_seconds: float
    summary_seconds: float


@dataclass
class JudgeResult:
    """Structured rubric-based score returned by the judge model."""

    overall: float
    coverage: float
    temporal_focus: float
    specificity: float
    clarity: float
    concision: float
    justification: str
    raw_response: str
    usage: dict[str, int | None]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the judge result."""
        payload = asdict(self)
        payload["overall"] = round(self.overall, 2)
        payload["coverage"] = round(self.coverage, 2)
        payload["temporal_focus"] = round(self.temporal_focus, 2)
        payload["specificity"] = round(self.specificity, 2)
        payload["clarity"] = round(self.clarity, 2)
        payload["concision"] = round(self.concision, 2)
        return payload


class OneDayTimelineEvaluator:
    """Runs one-day summary evaluations across present timeline agents."""

    def __init__(
        self,
        summary_model: str,
        judge_model: str = "google/gemini-3-flash-preview",
        judge_temperature: float = 0.0,
        summary_temperature: float = 0.2,
        base_url: str | None = None,
    ) -> None:
        """Initialize the evaluator with summariser and judge models."""
        if base_url is None:
            base_url = "https://openrouter.ai/api/v1"
        self.summary_model = summary_model
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.summary_temperature = summary_temperature
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def load_cases(self, path: Path) -> list[OneDayCase]:
        """Load evaluation cases from a JSONL file."""
        logging.info("Loading one-day evaluation cases path=%s", path)
        cases: list[OneDayCase] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                cases.append(OneDayCase.from_dict(payload))
        logging.info("Loaded one-day evaluation cases count=%s", len(cases))
        return cases

    def default_candidate_configs(
        self, selected: list[str] | None = None
    ) -> list[CandidateConfig]:
        """Return the default candidate configurations."""
        configs = [
            #CandidateConfig(
            #    candidate_id="perplexity_sonar",
            #    generator="perplexity",
            #    model="perplexity/sonar-deep-research",
            #    description="Perplexity Sonar research agent",
            #),
            CandidateConfig(
                candidate_id="simple_one_day",
                generator="simple_one_day",
                model="moonshotai/kimi-k2.5",
                description="Minimal one-day agent (Wikipedia + Exa tools, OpenAI chat)",
            ),
        ]
        if selected is None:
            return configs
        lookup = {config.candidate_id: config for config in configs}
        filtered = []
        for candidate_id in selected:
            filtered.append(lookup[candidate_id])
        return filtered

    def run(
        self,
        cases: list[OneDayCase],
        candidates: list[CandidateConfig],
        output_path: Path,
        rubric_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute evaluations and write JSONL records."""
        if rubric_text is None:
            raise ValueError(
                "rubric_text is required. Provide a rubric file via the harness CLI or pass a string here."
            )
        logging.info(
            "Starting one-day evaluations cases=%s candidates=%s output=%s",
            len(cases),
            [c.candidate_id for c in candidates],
            output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for case in cases:
            for config in candidates:
                record = self._evaluate_candidate(case, config, rubric_text)
                records.append(record)
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        logging.info(
            "Completed one-day evaluations records_written=%s path=%s",
            len(records),
            output_path,
        )
        return records

    def _evaluate_candidate(
        self, case: OneDayCase, config: CandidateConfig, rubric_text: str
    ) -> dict[str, Any]:
        """Run generation, summarisation, and judging for a single candidate."""
        logging.info(
            "Running candidate case_id=%s candidate=%s model=%s",
            case.case_id,
            config.candidate_id,
            config.model,
        )
        present_timeline, generator_metadata, elapsed_seconds = (
            self._generate_present_timeline(case, config)
        )
        summary, summary_usage, summary_seconds = self._summarise_one_day(
            present_timeline, case, config
        )
        judge_result = self._judge_summary(
            summary, case, config, rubric_text
        )
        return {
            "case_id": case.case_id,
            "topic": case.topic,
            "target_date": case.target_date,
            "candidate_id": config.candidate_id,
            "generator": {
                "type": config.generator,
                "model": config.model,
                "elapsed_seconds": elapsed_seconds,
                "metadata": generator_metadata,
            },
            "summary": summary,
            "summary_usage": summary_usage,
            "summary_seconds": summary_seconds,
            "judge": judge_result.to_dict(),
            "judge_usage": judge_result.usage,
            "judge_seconds": judge_result.elapsed_seconds,
        }

    def _generate_present_timeline(
        self, case: OneDayCase, config: CandidateConfig
    ) -> tuple[str, dict[str, Any], float]:
        """Generate a present timeline using the configured agent."""
        started_at = time.perf_counter()
        if config.generator == "dspy":
            output = generate_present_timeline(
                topic_pertaining_to=case.topic,
                time_until=case.target_date,
                model=config.model,
                target_timeline_chars=8000,
                min_events=6,
                max_events=12,
            )
            timeline = output["merged"]["merged_timeline"]
        elif config.generator == "perplexity":
            output = generate_present_timeline_perplexity(
                topic_pertaining_to=case.topic,
                current_date=case.target_date,
                model=config.model,
                temperature=0.2,
                max_tokens=8000,
            )
            timeline = output["timeline"]
        elif config.generator == "simple_wiki":
            output = generate_present_timeline_simple_wiki(
                topic_pertaining_to=case.topic,
                current_date=case.target_date,
                model=config.model,
                max_tokens=16000,
                max_iters=12,
            )
            timeline = output["timeline"]
        elif config.generator == "simple_one_day":
            output = generate_one_day_simple_agent(
                topic=case.topic,
                target_date=case.target_date,
                model=config.model,
                add_reasoning_content=True,
            )
            timeline = output["timeline"]
        else:
            raise ValueError(f"Unknown generator: {config.generator}")
        elapsed = time.perf_counter() - started_at
        logging.info(
            "Generated present timeline candidate=%s seconds=%.2f chars=%s",
            config.candidate_id,
            elapsed,
            len(timeline),
        )
        generator_metadata = {
            "model": config.model,
            "generator": config.generator,
            "elapsed_seconds": elapsed,
            "timeline_chars": len(timeline),
            "raw": output,
        }
        return timeline, generator_metadata, elapsed

    def _summarise_one_day(
        self, timeline: str, case: OneDayCase, config: CandidateConfig
    ) -> tuple[str, dict[str, int | None], float]:
        """Compress a multi-day timeline down to a single-day summary."""
        prompt = (
            "You are producing a one-day situation report.\n"
            f"Date of interest: {case.target_date}.\n"
            f"Topic: {case.topic}.\n"
            "Input is a longer present-day timeline. Extract only events that "
            "occurred on the date of interest or were reported on that date. "
            "Include concrete details (locations, casualty counts, figures, key quotes). "
            f"Write in markdown with a single heading '## {case.target_date}' followed by bullet points. "
            "Keep it under 180 words. If information for the date is missing, say "
            "'Insufficient same-day information' and nothing else."
        )
        started_at = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.summary_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": timeline},
            ],
            temperature=self.summary_temperature,
            max_tokens=700,
        )
        elapsed = time.perf_counter() - started_at
        message = completion.choices[0].message
        usage = _usage_to_dict(completion.usage)
        logging.info(
            "Summarised timeline candidate=%s seconds=%.2f tokens=%s",
            config.candidate_id,
            elapsed,
            usage["total_tokens"],
        )
        log_inference_call(
            name="one_day.summary",
            model=self.summary_model,
            inputs={
                "topic": case.topic,
                "target_date": case.target_date,
                "candidate_id": config.candidate_id,
                "prompt": prompt,
            },
            outputs={"summary": message.content},
            metadata={
                "case_id": case.case_id,
                "candidate_id": config.candidate_id,
                "usage": usage,
            },
            duration_seconds=elapsed,
        )
        return message.content, usage, elapsed

    def _judge_summary(
        self,
        summary: str,
        case: OneDayCase,
        config: CandidateConfig,
        rubric_text: str,
    ) -> JudgeResult:
        """Score a one-day summary using the provided rubric text."""
        rubric = (
            "You are an exacting evaluator of one-day news summaries.\n"
            "Use the rubric provided below as the sole grading guide. Do not invent new criteria.\n"
            "Return JSON with keys: coverage, temporal_focus, specificity, clarity, concision, overall, justification.\n"
            "All numeric scores must be 0-10 floats; overall should reflect the rubric weighting.\n"
            f"Rubric:\n{rubric_text}"
        )
        messages = [
            {
                "role": "system",
                "content": rubric,
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {case.topic}\nDate: {case.target_date}\nSummary to score:\n{summary}"
                ),
            },
        ]
        started_at = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.judge_model,
            messages=messages,
            temperature=self.judge_temperature,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        elapsed = time.perf_counter() - started_at
        message = completion.choices[0].message
        usage = _usage_to_dict(completion.usage)
        logging.info(
            "Judged summary candidate=%s seconds=%.2f tokens=%s",
            config.candidate_id,
            elapsed,
            usage["total_tokens"],
        )
        log_inference_call(
            name="one_day.judge",
            model=self.judge_model,
            inputs={
                "topic": case.topic,
                "target_date": case.target_date,
                "candidate_id": config.candidate_id,
                "rubric": rubric,
                "summary": summary,
            },
            outputs={"response": message.content},
            metadata={
                "case_id": case.case_id,
                "candidate_id": config.candidate_id,
                "usage": usage,
            },
            duration_seconds=elapsed,
        )
        scores = json.loads(message.content)
        return JudgeResult(
            overall=float(scores["overall"]),
            coverage=float(scores["coverage"]),
            temporal_focus=float(scores["temporal_focus"]),
            specificity=float(scores["specificity"]),
            clarity=float(scores["clarity"]),
            concision=float(scores["concision"]),
            justification=scores["justification"],
            raw_response=message.content,
            usage=usage,
            elapsed_seconds=elapsed,
        )
