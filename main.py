import gzip
import json
import logging
import os
from pathlib import Path

import fire
import more_itertools as mit
import tqdm
from dotenv import load_dotenv

import src.labelling_gen as labelling_gen
import src.news_enrichment as news_enrichment
import src.prediction_markets as prediction_markets
import src.prediction_markets_new as prediction_markets_new
import src.transcript_gen as transcript_gen
from src.news_data import NewsDownloader


def do_enrichment(p: Path, output: Path):
    if output.exists():
        return

    processors: list[news_enrichment.Processor] = [
        news_enrichment.LanguageIdentifier(),
        news_enrichment.MultilingualZeroShotClassifier(),
        news_enrichment.SeedTranslator(
            os.environ["hugging_face_hub_token"],
            os.environ["vllm_openai_api_key"],
            os.environ["vllm_openai_api_base"],
        ),
        news_enrichment.RestrictedHunyuanTranslator(
            os.environ["hugging_face_hub_token"],
            os.environ["vllm_openai_api_key"],
            os.environ["vllm_openai_api_base"],
        ),
        news_enrichment.TitleArbitrator(),
        # news_enrichment.ENPipelineClassifier("WebOrganizer/TopicClassifier-NoURL",trust_remote_code=True),
        # news_enrichment.ENPipelineClassifier("valurank/distilroberta-topic-classification"),
        # news_enrichment.ENPipelineClassifier("dstefa/roberta-base_topic_classification_nyt_news"),
        news_enrichment.ZeroShotClassifier(),
        # news_enrichment.Embedding(),
    ]

    with gzip.open(p, "rt") as i:
        articles = [json.loads(line) for line in i]

    for processor in processors:
        logging.info("Setup processor %s", processor)
        try:
            processor.setup()
            logging.info("Setup complete %s", processor)
            for articles_batch in mit.chunked(tqdm.tqdm(articles), 1):
                processor.process_batch(articles_batch)
        finally:
            logging.info("Shutdown processor %s", processor)
            processor.shutdown()

    with gzip.open(output, "wt", encoding="utf-8") as o:
        for article in articles:
            o.write(json.dumps(article))
            o.write("\n")


def redo_new_topic():
    articles = []
    ps = [
        Path.home() / "data/gdelt-gal-enriched/20250908100300.json.gz",
        Path.home() / "data/gdelt-gal-enriched/20250908100200.json.gz",
    ]
    for p in ps:
        with gzip.open(p, "rt") as i:
            articles.extend([json.loads(line) for line in i])

    processors = [news_enrichment.ZeroShotClassifier("energy resources and trade")]
    for processor in processors:
        logging.info("Setup processor %s", processor)
        try:
            processor.setup()
            logging.info("Setup complete %s", processor)
            for articles_batch in mit.chunked(tqdm.tqdm(articles), 1):
                processor.process_batch(articles_batch)
        finally:
            logging.info("Shutdown processor %s", processor)
            processor.shutdown()
    _test_generation(articles, ("energy_resources_and_trade", 0.85))


def _test_generation(enriched_articles: list[dict], topic_filter: tuple[str, float]):
    generator = transcript_gen.TranscriptGenerator(os.environ["openrouter_api_key"])

    topic, value = topic_filter

    articles = [
        d["title/arbitrated"] for d in enriched_articles if d.get(topic, 0) > value
    ]
    print(generator.generate(articles))


def test_generation(ps: list[Path], topic_filter: tuple[str, float]):
    enriched_articles = []
    for p in ps:
        with gzip.open(p, "rt") as i:
            enriched_articles.extend([json.loads(line) for line in i])
    _test_generation(enriched_articles, topic_filter)


def do_allpast_15():
    p = Path.home() / "data" / "gdelt-gal"
    o = Path.home() / "data" / "gdelt-gal-enriched"

    p.mkdir(exist_ok=True, parents=True)
    downloader = NewsDownloader(p)

    files = []
    logging.info("Downloading latest")
    latest_date, path = downloader.download_latest()
    logging.info("Downloaded latest %s %s", latest_date, path)
    files.append(path)
    d = latest_date.prev()
    for _ in range(14):
        logging.info("Trying download prev %s", d)
        try:
            path = downloader.download(d)
            files.append(path)
        except:
            pass
        d = d.prev()

    enriched_files = []
    for p in files:
        output_name = p.stem.split(".")[0] + ".json.gz"
        output_file = o / output_name
        do_enrichment(p, output_file)
        enriched_files.append(output_file)

    test_generation(enriched_files, ("zs_clf/international_politics", 0.85))


def redo():
    o = Path.home() / "data/gdelt-gal-enriched"
    test_generation(
        [o / "20250908133300.json.gz", o / "20250908133200.json.gz"],
        ("zs_clf/international_politics", 0.85),
    )


def get_rss_feeds():
    # zcat ~/data/gdelt-gal/*.gz | jq -r .domain | sort | uniq -c | sort -rnk1,1 > ~/data/gdelt-gal-domains.txt
    # with open(Path.home() / ~/data/gdelt-gal-domains.txt
    pass


def test_dspy():
    title_arbitrator = news_enrichment.TitleArbitrator()
    dspy_module = news_enrichment.ImpactRater(
        os.environ["hugging_face_hub_token"],
        os.environ["vllm_openai_api_key"],
        os.environ["vllm_openai_api_base"],
    )

    p = Path.home() / "data/gdelt-gal-enriched/20250904121800.json.gz"
    enriched_articles = []
    with gzip.open(p, "rt") as i:
        for line in i:
            enriched_articles.append(json.loads(line))

    try:
        dspy_module.setup()
        for a in enriched_articles:
            title_arbitrator.process(a)
            dspy_module.process(a)
            print(json.dumps(a))
    finally:
        dspy_module.shutdown()


def test_polymarket():
    processors = [
        news_enrichment.Embedding(),
        news_enrichment.PolymarketCompatScorer(prediction_markets.PolymarketApi()),
        news_enrichment.MarketRelevanceJudgement(
            os.environ["hugging_face_hub_token"],
            os.environ["vllm_openai_api_key"],
            os.environ["vllm_openai_api_base"],
        ),
    ]

    with gzip.open(
        Path.home() / "data/gdelt-gal-enriched/20250908100200.json.gz", "rt"
    ) as i:
        articles = [json.loads(line) for line in i][:700]

    for processor in processors:
        logging.info("Setup processor %s", processor)
        try:
            processor.setup()
            logging.info("Setup complete %s", processor)
            for articles_batch in mit.chunked(tqdm.tqdm(articles), 1):
                processor.process_batch(articles_batch)
        finally:
            logging.info("Shutdown processor %s", processor)
            processor.shutdown()

    with gzip.open(
        Path.home() / "data/gdelt-gal-market-predictions/20250908100200.json.gz",
        "wt",
        encoding="utf-8",
    ) as o:
        for article in articles:
            o.write(json.dumps(article))
            o.write("\n")


def classify_polymarket_geopolitics(
    input_path: Path | None = None,
    output_path: Path | None = None,
):
    data_dir = Path("data")
    if output_path is None:
        if input_path is None:
            output_path = data_dir / "polymarket_events_geopolitics.jsonl"
        else:
            output_path = input_path.with_name(f"{input_path.stem}_geopolitics.jsonl")

    classifier = news_enrichment.ZeroShotClassifier(class_name="geopolitics")
    classifier.setup()
    try:
        if input_path is None:
            markets = prediction_markets_new.PolymarketApi().get_open_markets()
        else:
            markets_data = []
            with open(input_path, "r", encoding="utf-8") as i:
                for line in tqdm.tqdm(i, desc="Loading markets"):
                    if not line.strip():
                        continue
                    markets_data.append(json.loads(line))
            markets = prediction_markets_new.OpenMarkets.from_api(markets_data)

        events = markets.group_by_event()
        event_groups = events.group_by_title_template()

        items = []
        for group in event_groups:
            descriptions: list[str] = []
            event_ids: list[str] = []
            event_slugs: list[str] = []
            market_ids: list[str] = []
            for event in group.events:
                event_ids.append(event.id)
                if event.slug:
                    event_slugs.append(event.slug)
                if event.description:
                    descriptions.append(event.description)
                for market in events.events.get(event, []):
                    market_ids.append(market.id)

            description = "\n\n".join(dict.fromkeys(descriptions))
            title = group.template_title
            text = f"{title}\n\n{description}".strip()
            group_id = " ".join(title.lower().split())
            items.append(
                {
                    "event_id": group_id,
                    "event_ids": event_ids,
                    "event_slugs": event_slugs,
                    "title": title,
                    "description": description,
                    "market_ids": market_ids,
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
                    o.write(
                        json.dumps(
                            {
                                "event_id": item["event_id"],
                                "event_ids": item["event_ids"],
                                "event_slugs": item["event_slugs"],
                                "event_description": item["description"],
                                "title": item["title"],
                                "description": item["description"],
                                "market_ids": item["market_ids"],
                                "probability": item.get("zs_clf/geopolitics", 0),
                            }
                        )
                    )
                    o.write("\n")
    finally:
        classifier.shutdown()


def output_eval_data_items():
    processors = [
        news_enrichment.Embedding(),
        news_enrichment.PolymarketCompatScorer(
            prediction_markets.PolymarketApi(), threshold=0
        ),
    ]

    with gzip.open(
        Path.home() / "data/gdelt-gal-enriched/20250910111700.json.gz", "rt"
    ) as i:
        articles = [json.loads(line) for line in i][:2000]

    for processor in processors:
        logging.info("Setup processor %s", processor)
        try:
            processor.setup()
            logging.info("Setup complete %s", processor)
            for articles_batch in mit.chunked(tqdm.tqdm(articles), 1):
                processor.process_batch(articles_batch)
        finally:
            logging.info("Shutdown processor %s", processor)
            processor.shutdown()

    labelling_generator = labelling_gen.LabelingGenerator(
        os.environ["openrouter_api_key"], model="openrouter/sonoma-dusk-alpha"
    )
    for el in tqdm.tqdm(articles):
        if len(el.get("polymarket/relevant_market_candidates", [])) == 0:
            continue
        max_market = max(
            el["polymarket/relevant_market_candidates"], key=lambda x: x["similarity"]
        )["market"]
        d = {
            "news_headline": el["title/arbitrated"],
            "market_event": max_market["question"],
            "market_event_description": max_market["description"],
            "news_date": el["date"][:10],
            "news_domain": el["domain"],
            "news_outlet_name": el["outletName"],
        }
        try:
            label = labelling_generator.generate(d)
            print(json.dumps({"input": d, "output": label}))
        except Exception as e:
            logging.exception(e)


def test_cross_scorer():
    processors = [
        news_enrichment.Embedding(),
        news_enrichment.PolymarketCompatScorer(
            prediction_markets.PolymarketApi(), threshold=0.42
        ),
        news_enrichment.PolymarketCompatCrossScorer(
            prediction_markets.PolymarketApi(), threshold=0
        ),
    ]
    with gzip.open(
        Path.home() / "data/gdelt-gal-enriched/20250910111600.json.gz", "rt"
    ) as i:
        articles = [json.loads(line) for line in i][:100]

    for processor in processors:
        processor.setup()

        try:
            for article in tqdm.tqdm(articles):
                processor.process(article)
        finally:
            processor.shutdown()

    for a in articles:
        print(json.dumps(a))


if __name__ == "__main__":
    load_dotenv()  # take environment variables
    logging.basicConfig(level=logging.DEBUG)
    fire.Fire()
