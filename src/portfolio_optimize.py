from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from src.portfolio_optimization import PortfolioOptimizationConfig, PortfolioOptimizer
from src.prediction_markets import EventGroup, Events


def load_latest_market_probabilities(path: Path) -> dict[str, dict[str, dict]]:
    """Load the latest probability record per market id."""
    start_time = time.time()
    logging.info("Loading market probabilities from %s", path)
    records = path.read_text(encoding="utf-8").splitlines()
    grouped: dict[str, dict[str, dict]] = {}
    for line in records:
        record = json.loads(line)
        event_group_id = record["event_group_id"]
        market_id = record["market_id"]
        if event_group_id not in grouped:
            grouped[event_group_id] = {}
        if market_id not in grouped[event_group_id]:
            grouped[event_group_id][market_id] = record
            continue
        if record["generated_at"] > grouped[event_group_id][market_id]["generated_at"]:
            grouped[event_group_id][market_id] = record
    elapsed = time.time() - start_time
    logging.info(
        "Loaded %d event groups in %.2fs from market probabilities",
        len(grouped),
        elapsed,
    )
    return grouped


def load_event_group_index(events_path: Path) -> dict[str, EventGroup]:
    """Load event groups from events.jsonl and index by event_group_id."""
    start_time = time.time()
    logging.info("Loading event groups from %s", events_path)
    events = Events.from_file(events_path)
    event_groups = {group.id(): group for group in events.group_by_title_template()}
    elapsed = time.time() - start_time
    logging.info("Loaded %d event groups in %.2fs", len(event_groups), elapsed)
    return event_groups


def build_optimizer_inputs(
    event_group_index: dict[str, EventGroup],
    probability_index: dict[str, dict[str, dict]],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Build optimizer inputs from latest probabilities and open market prices."""
    start_time = time.time()
    logging.info("Building optimizer inputs from open markets")
    market_rows: list[dict] = []
    p_yes_values: list[float] = []
    q_values: list[float] = []
    for event_group_id, markets in probability_index.items():
        event_group = event_group_index[event_group_id]
        market_lookup = {
            market.id: market
            for market in event_group.open_markets()
            if market.active and not market.closed
        }
        for market_id, record in markets.items():
            market = market_lookup[market_id]
            p_yes = market.yes_probability()
            if p_yes is None:
                raise ValueError(f"Market {market_id} missing YES probability.")
            q = record["probability"]
            market_rows.append(
                {
                    "event_group_id": event_group_id,
                    "market_id": market_id,
                    "market_question": record["market_question"],
                    "market_slug": record["market_slug"],
                    "p_yes": p_yes,
                    "q": q,
                }
            )
            p_yes_values.append(p_yes)
            q_values.append(q)
    elapsed = time.time() - start_time
    logging.info(
        "Built optimizer inputs for %d markets in %.2fs",
        len(market_rows),
        elapsed,
    )
    return market_rows, np.array(p_yes_values, dtype=float), np.array(q_values, dtype=float)


def calculate_marginal_gains(
    p_yes: float,
    q: float,
    alloc: float,
) -> tuple[float, float, float]:
    """Calculate marginal gains for YES, NO, and the current allocation direction."""
    p_no = 1 - p_yes
    if alloc > 0:
        f_yes = alloc
        f_no = 0.0
    elif alloc < 0:
        f_yes = 0.0
        f_no = -alloc
    else:
        f_yes = 0.0
        f_no = 0.0

    w_yes = 1 + f_yes * (1 - p_yes) / p_yes - f_no
    w_no = 1 + f_no * (1 - p_no) / p_no - f_yes

    marginal_yes = q * ((1 - p_yes) / p_yes) / w_yes + (1 - q) * (-1) / w_no
    marginal_no = q * (-1) / w_yes + (1 - q) * ((1 - p_no) / p_no) / w_no

    if alloc > 0:
        directional = marginal_yes
    elif alloc < 0:
        directional = marginal_no
    else:
        directional = max(marginal_yes, marginal_no)

    return marginal_yes, marginal_no, directional


def run_portfolio_optimizer(
    probabilities_path: Path,
    events_path: Path,
    max_fraction_per_market: float,
    max_total_fraction: float,
    kelly_fraction: float = 1.0,
    turnover_penalty: float = 0.0,
    epsilon: float = 1e-9,
    solver: str = "ECOS",
    round_threshold: float = 1e-6,
) -> dict:
    """Run the portfolio optimizer from latest probabilities and market prices."""
    probability_index = load_latest_market_probabilities(probabilities_path)
    event_group_index = load_event_group_index(events_path)
    market_rows, p_yes, q = build_optimizer_inputs(event_group_index, probability_index)

    if len(p_yes) == 0:
        raise ValueError("No open markets found with probabilities.")

    optimizer = PortfolioOptimizer(
        PortfolioOptimizationConfig(
            max_fraction_per_market=max_fraction_per_market,
            max_total_fraction=max_total_fraction,
            kelly_fraction=kelly_fraction,
            turnover_penalty=turnover_penalty,
            epsilon=epsilon,
            solver=solver,
        )
    )

    current_alloc = np.zeros(len(p_yes), dtype=float)
    result = optimizer.optimize(
        p_yes=p_yes,
        q=q,
        current_alloc=current_alloc,
    )

    allocations = []
    for row, raw_alloc in zip(market_rows, result.target_alloc, strict=True):
        marginal_yes, marginal_no, directional = calculate_marginal_gains(
            row["p_yes"],
            row["q"],
            raw_alloc,
        )
        alloc = raw_alloc
        if abs(alloc) < round_threshold:
            alloc = 0.0
        allocations.append(
            {
                "event_group_id": row["event_group_id"],
                "market_id": row["market_id"],
                "market_question": row["market_question"],
                "market_slug": row["market_slug"],
                "p_yes": row["p_yes"],
                "q": row["q"],
                "target_alloc": alloc,
                "marginal_gain_yes": marginal_yes,
                "marginal_gain_no": marginal_no,
                "marginal_gain_directional": directional,
            }
        )

    return {
        "expected_log_growth": result.expected_log_growth,
        "allocations": allocations,
    }
