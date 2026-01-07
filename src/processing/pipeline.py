import datetime as dt
import json
import logging
from pathlib import Path

import more_itertools as mit
import tqdm

from ..news_enrichment import ZeroShotClassifier
from ..prediction_markets import Events, PolymarketApi
from ..news_data import NewsDownloader


def fetch_polymarkets_events(
    output_path=Path("data")
    / f"polymarket_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
):
    api = PolymarketApi()
    with output_path.open("w", encoding="utf-8") as handle:
        for event in api.iter_event_dicts():
            print(json.dumps(event, ensure_ascii=False), file=handle)
    logging.info(f"Wrote {output_path}")


def classify_event_geopol_prob(output_path: Path, polymarkets_events_path: Path):
    events = Events.from_file(polymarkets_events_path)
    event_groups = events.group_by_title_template()

    classifier = ZeroShotClassifier(class_name="geopolitics")
    classifier.setup()
    try:
        items = []
        for group in event_groups:
            descriptions: list[str] = []
            for event in group.events:
                if event.description:
                    descriptions.append(event.description)

            description = "\n\n".join(dict.fromkeys(descriptions))
            title = group.template_title
            text = f"{title}\n\n{description}".strip()

            items.append(
                {
                    "id": group.id(),
                    "title/arbitrated": text,
                }
            )

        with open(output_path, "w", encoding="utf-8") as o:
            for batch_items in mit.chunked(
                tqdm.tqdm(items, desc="Classifying events"), 100
            ):
                if not batch_items:
                    continue
                classifier.process_batch(batch_items)
                for item in batch_items:
                    print(
                        json.dumps(
                            {
                                "id": item["id"],
                                "probability": item.get("zs_clf/geopolitics", 0),
                            }
                        ),
                        file=o,
                    )
        logging.info(f"Wrote {output_path}")
    finally:
        classifier.shutdown()


def generate_event_table(
    output_path: Path,
    polymarket_geopol_prob_path: Path,
    polymarket_events_path: Path,
):
    event_geopol_prob = dict[str, float]()
    with open(polymarket_geopol_prob_path) as i:
        for line in i:
            d = json.loads(line)
            event_geopol_prob[d["id"]] = d["probability"]

    events = Events.from_file(polymarket_events_path)
    events_groups = events.group_by_title_template()
    with output_path.open("w") as o:
        for event_group in events_groups:
            event_group_open_markets = list(event_group.open_markets())
            total_liq = sum([m.liquidity for m in event_group_open_markets if m.liquidity is not None], 0)
            total_vol = sum([m.volume for m in event_group_open_markets if m.volume is not None], 0)
            num_open_markets = len(event_group_open_markets)
            event_urls = "\n".join([e.url() for e in event_group.events])
            event_names = "\n".join([e.title for e in event_group.events])
            yes_probs: list[float] = list(filter(None, [m.yes_probability() for m in event_group_open_markets]))
            max_uncertainty = max(4*yes_prob*(1 - yes_prob) for yes_prob in yes_probs) if yes_probs else None
            event_group_id = event_group.id()
            event_group_geopol_prob = event_geopol_prob.get(event_group_id)
            print(
                json.dumps(
                    {
                        "event_group_id": event_group_id,
                        "total_liq": total_liq,
                        "total_vol": total_vol,
                        "num_open_markets": num_open_markets,
                        "event_urls": event_urls,
                        "event_names": event_names,
                        "max_uncertainty": max_uncertainty,
                        "event_group_geopol_prob": event_group_geopol_prob
                    }
                ),
                file=o
            )
    logging.info(f"Wrote {output_path}")


def download_news_past_week(news_base_path):
    for _ in NewsDownloader(news_base_path).download_past_week():
        pass