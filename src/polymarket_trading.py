from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType


@dataclass(frozen=True)
class PolymarketCredentials:
    """Credential bundle for Polymarket API and signer."""

    api_key: str
    api_secret: str
    api_passphrase: str
    private_key: str
    funder: str

    @classmethod
    def from_env(cls) -> "PolymarketCredentials":
        """Load Polymarket credentials from environment variables."""
        import os

        return cls(
            api_key=os.environ["POLYMARKET_API_KEY"],
            api_secret=os.environ["POLYMARKET_API_SECRET"],
            api_passphrase=os.environ["POLYMARKET_API_PASSPHRASE"],
            private_key=os.environ["POLYMARKET_PRIVATE_KEY"],
            funder=os.environ["POLYMARKET_FUNDER"],
        )


@dataclass(frozen=True)
class PolymarketTradeConfig:
    """Configuration for trading via the Polymarket CLOB."""

    bankroll_usd: float
    host: str = "https://clob.polymarket.com"
    chain_id: int = 137
    dry_run: bool = False


@dataclass(frozen=True)
class MarketMetadata:
    """Resolved metadata required to trade a Polymarket market."""

    market_id: str
    question: str
    outcome_tokens: dict[str, str]
    outcome_prices: dict[str, float]
    price_tick_size: float
    min_size: float
    accepting_orders: bool


@dataclass(frozen=True)
class TradeOrder:
    """Trade order derived from a target allocation."""

    market_id: str
    token_id: str
    outcome: str
    side: str
    price: float
    size: float
    notional_usd: float
    question: str


def round_price_to_tick(price: float, tick_size: float) -> float:
    """Round a price to the nearest tick size."""
    if tick_size <= 0:
        raise ValueError(f"Invalid tick size: {tick_size}")
    ticks = round(price / tick_size)
    return max(tick_size, ticks * tick_size)


class PolymarketTrader:
    """Place Polymarket CLOB orders from optimizer allocations."""

    def __init__(
        self,
        credentials: PolymarketCredentials,
        config: PolymarketTradeConfig,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the trader with credentials and config."""
        self.credentials = credentials
        self.config = config
        self.session = session or requests.Session()
        self.client = (
            None
            if config.dry_run
            else ClobClient(
                config.host,
                chain_id=config.chain_id,
                key=credentials.private_key,
                creds=ApiCreds(
                    api_key=credentials.api_key,
                    api_secret=credentials.api_secret,
                    api_passphrase=credentials.api_passphrase,
                ),
                funder=credentials.funder,
            )
        )

    def fetch_market_metadata(self, market_ids: set[str]) -> dict[str, MarketMetadata]:
        """Fetch market metadata for the provided market ids."""
        start_time = time.time()
        logging.info("Fetching Polymarket metadata for %d markets", len(market_ids))
        metadata: dict[str, MarketMetadata] = {}
        for market_id in market_ids:
            response = self.session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"id": market_id},
            )
            response.raise_for_status()
            market_data = response.json()[0]
            outcomes = json.loads(market_data["outcomes"])
            token_ids = json.loads(market_data["clobTokenIds"])
            outcome_prices = json.loads(market_data["outcomePrices"])
            outcome_tokens = dict(zip(outcomes, token_ids, strict=True))
            outcome_prices_map = {
                outcome: float(price)
                for outcome, price in zip(outcomes, outcome_prices, strict=True)
            }
            metadata[market_id] = MarketMetadata(
                market_id=str(market_data["id"]),
                question=str(market_data["question"]),
                outcome_tokens=outcome_tokens,
                outcome_prices=outcome_prices_map,
                price_tick_size=float(market_data["orderPriceMinTickSize"]),
                min_size=float(market_data["orderMinSize"]),
                accepting_orders=bool(market_data["acceptingOrders"]),
            )
        elapsed = time.time() - start_time
        logging.info(
            "Fetched metadata for %d markets in %.2fs",
            len(metadata),
            elapsed,
        )
        return metadata

    def build_trade_orders(
        self,
        allocations: list[dict],
        market_metadata: dict[str, MarketMetadata],
        min_abs_alloc: float,
    ) -> list[TradeOrder]:
        """Convert target allocations into executable order specs."""
        start_time = time.time()
        logging.info("Building trade orders from %d allocations", len(allocations))
        orders: list[TradeOrder] = []
        for allocation in allocations:
            target_alloc = allocation["target_alloc"]
            if abs(target_alloc) < min_abs_alloc:
                continue
            market_id = allocation["market_id"]
            metadata = market_metadata[market_id]
            if not metadata.accepting_orders:
                raise ValueError(f"Market {market_id} is not accepting orders.")
            if target_alloc > 0:
                outcome = "Yes"
                price = metadata.outcome_prices[outcome]
            else:
                outcome = "No"
                price = metadata.outcome_prices[outcome]
            price = round_price_to_tick(price, metadata.price_tick_size)
            notional_usd = self.config.bankroll_usd * abs(target_alloc)
            size = notional_usd / price
            size = math.floor(size * 1e6) / 1e6
            if size < metadata.min_size:
                logging.info(
                    "Skipping market %s due to min size %.4f < %.4f",
                    market_id,
                    size,
                    metadata.min_size,
                )
                continue
            orders.append(
                TradeOrder(
                    market_id=market_id,
                    token_id=metadata.outcome_tokens[outcome],
                    outcome=outcome,
                    side="BUY",
                    price=price,
                    size=size,
                    notional_usd=notional_usd,
                    question=metadata.question,
                )
            )
        elapsed = time.time() - start_time
        logging.info(
            "Built %d trade orders in %.2fs",
            len(orders),
            elapsed,
        )
        return orders

    def place_orders(self, orders: list[TradeOrder]) -> list[dict]:
        """Submit orders to the Polymarket CLOB."""
        start_time = time.time()
        logging.info("Placing %d Polymarket orders", len(orders))
        results: list[dict] = []
        for order in orders:
            logging.info(
                "Placing order market=%s outcome=%s price=%.4f size=%.4f notional=%.2f",
                order.market_id,
                order.outcome,
                order.price,
                order.size,
                order.notional_usd,
            )
            if self.config.dry_run:
                results.append({"market_id": order.market_id, "status": "dry_run"})
                continue
            order_args = OrderArgs(
                token_id=order.token_id,
                price=order.price,
                size=order.size,
                side=order.side,
            )
            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, orderType=OrderType.GTC)
            results.append(response)
        elapsed = time.time() - start_time
        logging.info("Placed %d orders in %.2fs", len(results), elapsed)
        return results
