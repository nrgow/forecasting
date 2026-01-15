import logging
import time
from dataclasses import dataclass, field, asdict

import dspy
import datetime as dt

# import src.gdelt_api as gdelt_api
from . import tools
from ..mlflow_tracing import configure_dspy_autolog, log_inference_call


@dataclass
class Event:
    date: str
    description: str
    source: str | None


@dataclass
class EventStore:
    events: list[Event] = field(default_factory=list)

    def add_event(self, event: Event):
        self.events.append(event)


class ExtractEvents(dspy.Signature):
    """Extract dated events from a timeline draft within requested bounds."""

    topic_pertaining_to: str = dspy.InputField()
    content: str = dspy.InputField()
    min_events: int = dspy.InputField()
    max_events: int = dspy.InputField()
    events: list[Event] = dspy.OutputField()


class EventsToTimeline(dspy.Signature):
    """Convert structured events into a narrative timeline."""

    topic_pertaining_to: str = dspy.InputField()
    events: list[Event] = dspy.InputField()
    timeline: str = dspy.OutputField()


class EventsTimeline(dspy.Signature):
    """Draft a detailed present-day timeline for the topic."""

    topic_pertaining_to: str = dspy.InputField()
    time_until: str = dspy.InputField()
    target_length_chars: int = dspy.InputField()
    timeline: str = dspy.OutputField()


class SpecificEventsTimeline(dspy.Signature):
    """Expand a specific subtopic into a detailed timeline."""

    topic_pertaining_to: str = dspy.InputField()
    subtopic_pertaining_to: str = dspy.InputField()
    date: str = dspy.InputField()
    target_length_chars: int = dspy.InputField()
    timeline: str = dspy.OutputField()


@dataclass
class Subtimeline:
    subtopic_pertaining_to: str
    date: str
    subtimeline: str


class MergeTimelines(dspy.Signature):
    """Merge multiple subtimelines into a cohesive chronology."""

    overall_topic_pertaining_to: str = dspy.InputField()
    subtimelines: list[Subtimeline] = dspy.InputField()
    target_length_chars: int = dspy.InputField()
    merged_timeline: str = dspy.OutputField(
        desc="The merged timeline containing all the information in the subtimelines but arranged chronologically and narrative flow"
    )


def generate_present_timeline(
    topic_pertaining_to: str,
    time_until=str(dt.date.today()),
    model="openrouter/x-ai/grok-4.1-fast",
    target_timeline_chars: int = 24000,
    min_events: int = 18,
    max_events: int = 28,
    event_group_id: str | None = None,
    run_id: str | None = None,
):
    """Generate a present timeline with enough detail for long-context usage."""
    configure_dspy_autolog()
    dspy.configure(lm=dspy.LM(model))

    wiki = tools.CachedWikipedia()
    # news_api = gdelt_api.GDELTDocAPI()
    first_timeline_target_chars = max(6000, target_timeline_chars // 3)
    subtimeline_target_chars = max(1200, target_timeline_chars // min_events)

    loop = dspy.ReAct(
        EventsTimeline,
        tools=[
            wiki.get_wikipedia_page,
            wiki.search_wikipedia_pages,
            # news_api.news_search,
            # tools.fetch_webpage_content,
        ],
        max_iters=8,
    )

    logging.info("Present timeline draft starting topic=%s", topic_pertaining_to)
    first_started_at = time.perf_counter()
    first_timeline = loop(
        topic_pertaining_to=topic_pertaining_to,
        time_until=time_until,
        target_length_chars=first_timeline_target_chars,
    )
    first_elapsed = time.perf_counter() - first_started_at
    logging.info("Present timeline draft completed seconds=%.2f", first_elapsed)
    log_inference_call(
        name="present_timeline.draft",
        model=model,
        inputs={
            "topic_pertaining_to": topic_pertaining_to,
            "time_until": time_until,
            "target_length_chars": first_timeline_target_chars,
        },
        outputs={"timeline": first_timeline.timeline},
        metadata={
            "pipeline_run_id": run_id,
            "event_group_id": event_group_id,
            "stage": "draft",
        },
        duration_seconds=first_elapsed,
    )

    timeline_extraction = dspy.Predict(ExtractEvents)

    logging.info("Present timeline extraction starting topic=%s", topic_pertaining_to)
    extraction_started_at = time.perf_counter()
    extracted_events = timeline_extraction(
        topic_pertaining_to=topic_pertaining_to,
        content=first_timeline.timeline,
        min_events=min_events,
        max_events=max_events,
    )
    extraction_elapsed = time.perf_counter() - extraction_started_at
    logging.info(
        "Present timeline extraction completed seconds=%.2f", extraction_elapsed
    )
    log_inference_call(
        name="present_timeline.extract_events",
        model=model,
        inputs={
            "topic_pertaining_to": topic_pertaining_to,
            "content": first_timeline.timeline,
            "min_events": min_events,
            "max_events": max_events,
        },
        outputs={
            "events": [asdict(event) for event in extracted_events.events],
        },
        metadata={
            "pipeline_run_id": run_id,
            "event_group_id": event_group_id,
            "stage": "extract_events",
        },
        duration_seconds=extraction_elapsed,
    )

    sub_loop = dspy.ReAct(
        SpecificEventsTimeline,
        tools=[
            wiki.get_wikipedia_page,
            wiki.search_wikipedia_pages,
            # news_api.news_search,
            # tools.fetch_webpage_content,
        ],
        max_iters=5,
    )

    subtimelines = []
    for event in extracted_events.events:
        logging.info("Generating sub timeline for %s", event.description)
        sub_started_at = time.perf_counter()
        subtimeline = sub_loop(
            topic_pertaining_to=topic_pertaining_to,
            subtopic_pertaining_to=event.description,
            date=event.date,
            target_length_chars=subtimeline_target_chars,
        )
        sub_elapsed = time.perf_counter() - sub_started_at
        log_inference_call(
            name="present_timeline.subtimeline",
            model=model,
            inputs={
                "topic_pertaining_to": topic_pertaining_to,
                "subtopic_pertaining_to": event.description,
                "date": event.date,
                "target_length_chars": subtimeline_target_chars,
            },
            outputs={"timeline": subtimeline.timeline},
            metadata={
                "pipeline_run_id": run_id,
                "event_group_id": event_group_id,
                "stage": "subtimeline",
            },
            duration_seconds=sub_elapsed,
        )
        subtimelines.append(subtimeline)

    merge = dspy.ChainOfThought(MergeTimelines)
    logging.info("Present timeline merge starting topic=%s", topic_pertaining_to)
    merge_started_at = time.perf_counter()
    merged = merge(
        overall_topic_pertaining_to=topic_pertaining_to,
        subtimelines=[
            Subtimeline(
                subtopic_pertaining_to=event.description,
                date=event.date,
                subtimeline=st.timeline,
            )
            for event, st in zip(extracted_events.events, subtimelines)
        ],
        target_length_chars=target_timeline_chars,
    )
    merge_elapsed = time.perf_counter() - merge_started_at
    logging.info("Present timeline merge completed seconds=%.2f", merge_elapsed)
    log_inference_call(
        name="present_timeline.merge",
        model=model,
        inputs={
            "overall_topic_pertaining_to": topic_pertaining_to,
            "subtimelines": [
                {
                    "subtopic_pertaining_to": event.description,
                    "date": event.date,
                    "subtimeline": st.timeline,
                }
                for event, st in zip(extracted_events.events, subtimelines)
            ],
            "target_length_chars": target_timeline_chars,
        },
        outputs={"merged_timeline": merged.merged_timeline},
        metadata={
            "pipeline_run_id": run_id,
            "event_group_id": event_group_id,
            "stage": "merge",
        },
        duration_seconds=merge_elapsed,
    )
    output = {
        "first_timeline": first_timeline.toDict(),
        "extracted_events": extracted_events.toDict(),
        "subtimelines": [st.toDict() for st in subtimelines],
        "merged": merged.toDict(),
        "target_timeline_chars": target_timeline_chars,
    }
    output["extracted_events"]["events"] = [
        asdict(e) for e in output["extracted_events"]["events"]
    ]
    return output
