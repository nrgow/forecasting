import datetime as dt
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import requests
from requests import Session
from requests_cache import CacheMixin
from requests_ratelimiter import LimiterMixin
from slugify import slugify


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    """Session class with caching and rate-limiting behavior.
    Accepts keyword arguments for both LimiterSession and CachedSession.
    """


class NoCacheLimiterSession(LimiterMixin, Session):
    """Session class with rate-limiting behavior."""


@dataclass(frozen=True)
class OpenMarket:
    id: str
    question: str
    slug: str
    liquidity: float | None
    volume: float | None
    volume_1wk: float | None
    end_date: str | None
    active: bool
    closed: bool
    outcomes: list[str]
    outcome_prices: list[float]

    @classmethod
    def from_api(cls, data: dict) -> "OpenMarket":
        """Build an OpenMarket from Polymarket market data."""
        if "liquidityNum" in data:
            liquidity = data["liquidityNum"]
        elif "liquidity" in data:
            liquidity = float(data["liquidity"])
        else:
            liquidity = None
        if "volumeNum" in data:
            volume = data["volumeNum"]
        elif "volume" in data:
            volume = float(data["volume"])
        else:
            volume = None
        if "volume1wk" in data:
            volume_1wk = float(data["volume1wk"])
        else:
            volume_1wk = None
        # event = MarketEvent.from_api(data["event"])
        return cls(
            id=str(data.get("id", "")),
            question=str(data.get("question", "")),
            slug=str(data.get("slug", "")),
            liquidity=liquidity,
            volume=volume,
            volume_1wk=volume_1wk,
            end_date=data.get("endDateIso") or data.get("endDate"),
            active=bool(data.get("active", False)),
            closed=bool(data.get("closed", False)),
            outcomes=json.loads(data["outcomes"]),
            outcome_prices=[
                float(x) for x in json.loads(data.get("outcomePrices", "[]"))
            ],
        )

    def yes_probability(self) -> float | None:
        if len(self.outcomes) != len(self.outcome_prices):
            return None
        try:
            yes_index = self.outcomes.index("Yes")
        except ValueError:
            return None
        return self.outcome_prices[yes_index]


@dataclass(frozen=True)
class Event:
    id: str
    slug: str
    title: str
    description: str | None
    start_date: str | None
    end_date: str | None
    open_markets: list[OpenMarket]

    @classmethod
    def from_api(cls, data: dict) -> "Event":
        return cls(
            id=str(data.get("id", "")),
            slug=str(data.get("slug", "")),
            title=str(data.get("title", "")),
            description=data.get("description"),
            start_date=data.get("startDate"),
            end_date=data.get("endDate"),
            open_markets=[OpenMarket.from_api(m) for m in data["markets"]],
        )

    def url(self):
        return f"https://polymarket.com/event/{self.slug}"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return False
        return self.id == other.id


def parse_open_market_end_date(value: str) -> dt.datetime:
    """Parse an open market end date into a timezone-aware datetime."""
    if "T" in value:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.datetime.combine(
        dt.date.fromisoformat(value),
        dt.time.min,
        tzinfo=dt.timezone.utc,
    )


class Events(list[Event]):
    @classmethod
    def from_file(cls, p: Path):
        events = []
        with open(p) as i:
            for line in i:
                events.append(Event.from_api(json.loads(line)))
        return Events(events)

    def group_by_title_template(
        self,
    ) -> "EventGroups":
        grouped: dict[str, list[Event]] = defaultdict(list)
        template_titles: dict[str, str] = {}
        for event in self:
            template = _title_template(event.title)
            key = _template_key(template)
            grouped[key].append(event)
            template_titles[key] = template
        return EventGroups(
            [
                EventGroup(template_title=template_titles[key], events=grouped[key])
                for key in grouped
            ]
        )


_MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
)
_QUARTER_PATTERN = r"q[1-4](?:\s+\d{4})?"
_END_OF_PATTERN = (
    rf"(?:the\s+)?end\s+of\s+"
    rf"(?:"
    rf"(?:{_MONTH_PATTERN})(?:\s+\d{{4}})?"
    rf"|\d{{4}}"
    rf"|{_QUARTER_PATTERN}"
    rf"|\d{{4}}\s+[a-zA-Z ]*season"
    rf")"
)
_DATE_PATTERN = (
    rf"(?:{_MONTH_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?"
    rf"|(?:{_MONTH_PATTERN})(?:,?\s*\d{{4}})?"
    rf"|\d{{1,2}}[/-]\d{{1,2}}(?:[/-]\d{{2,4}})?"
    rf"|{_QUARTER_PATTERN}"
    rf"|\d{{4}}"
    rf"|{_END_OF_PATTERN}"
)
_BY_BEFORE_DATE_RE = re.compile(
    rf"\b(?:by|before)\b\s+(?:{_DATE_PATTERN})(?:\b|[^a-zA-Z]).*$",
    re.IGNORECASE,
)
_PREP_DATE_RE = re.compile(
    rf"\b(?:on|in|as\s+of)\b\s+(?:{_DATE_PATTERN})(?:\b|[^a-zA-Z]).*$",
    re.IGNORECASE,
)


def _title_template(title: str) -> str:
    stripped = title.strip()
    match = _BY_BEFORE_DATE_RE.search(stripped)
    if match:
        return stripped[: match.start()].rstrip(" ?:-").strip()
    match = _PREP_DATE_RE.search(stripped)
    if match:
        return stripped[: match.start()].rstrip(" ?:-").strip()
    return stripped


def _template_key(template_title: str) -> str:
    return " ".join(template_title.lower().split())


@dataclass
class EventGroup:
    template_title: str
    events: list[Event]

    def id(self):
        return slugify(self.template_title)

    def open_markets(self) -> Iterable["OpenMarket"]:
        """Yield all open markets in the event group."""
        for event in self.events:
            yield from event.open_markets

    def latest_open_market_end_date(self) -> dt.datetime | None:
        """Return the latest end date across open markets."""
        end_dates = [
            parse_open_market_end_date(market.end_date)
            for market in self.open_markets()
            if market.end_date is not None
        ]
        return max(end_dates) if end_dates else None


class EventGroups(list[EventGroup]):
    pass


@dataclass
class PolymarketApi:
    url: str = "https://gamma-api.polymarket.com/events"
    session: requests.Session = field(
        default_factory=lambda: CachedLimiterSession(
            cache_name=f"requests-cache/polymarkets_requests_cache_{dt.date.today().strftime('%Y-%m-%d')}",
            per_second=5,
        )
    )

    def iter_event_dicts(self) -> Iterable[dict]:
        n = 20
        i = 0
        while True:
            response = self.session.get(
                self.url,
                params={
                    "limit": n,
                    "offset": i,
                    "active": "true",
                    "closed": "false",
                },
            )
            results = response.json()
            if len(results) == 0:
                return
            for event in results:
                event_info = {
                    "id": event.get("id"),
                    "slug": event.get("slug"),
                    "title": event.get("title"),
                    "description": event.get("description"),
                    "startDate": event.get("startDate"),
                    "endDate": event.get("endDate"),
                    "markets": [m for m in event.get("markets", []) if m["active"]],
                }
                yield event_info

            i += n
