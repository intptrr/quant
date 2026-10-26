from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from typing import Any

import httpx
import polars as pl

from quant.data.providers.sharadar import SharadarProvider
from quant.execution.broker import BrokerOrder, OrderRejectedError

KEY = "secret-key"


class FakeBroker:
    def __init__(self, equity: float = 100_000.0) -> None:
        self.account_equity = equity
        self.held: dict[str, float] = {}
        self.orders: dict[str, BrokerOrder] = {}
        self.borrowable: set[str] = set()
        self.refusals: dict[str, str] = {}
        self.closed = False

    def equity(self) -> float:
        return self.account_equity

    def positions(self) -> dict[str, float]:
        return {symbol: quantity for symbol, quantity in self.held.items() if quantity}

    def open_orders(self) -> list[BrokerOrder]:
        return [order for order in self.orders.values() if not order.done]

    def order(self, client_order_id: str) -> BrokerOrder | None:
        return self.orders.get(client_order_id)

    def submit_opening_order(self, client_order_id: str, symbol: str, quantity: int) -> BrokerOrder:
        if client_order_id in self.orders:
            raise ValueError(f"duplicate client order id {client_order_id}")
        if symbol in self.refusals:
            raise OrderRejectedError(self.refusals[symbol])
        order = BrokerOrder(client_order_id, symbol, float(quantity), 0.0, None, "accepted")
        self.orders[client_order_id] = order
        return order

    def shortable(self, symbols: Sequence[str]) -> dict[str, bool]:
        return {symbol: symbol in self.borrowable for symbol in symbols}

    def close(self) -> None:
        self.closed = True

    def open_auction(self, prices: Mapping[str, float], commission: float | None = None) -> None:
        for client_order_id, order in self.orders.items():
            if order.done:
                continue
            self.orders[client_order_id] = replace(
                order,
                filled_quantity=order.quantity,
                filled_price=prices[order.symbol],
                status="filled",
                commission=commission,
            )
            self.held[order.symbol] = self.held.get(order.symbol, 0.0) + order.quantity


class FakeMarket:
    def __init__(self) -> None:
        self.prices: dict[str, float] = {}
        self.announced_splits: dict[tuple[str, date], str] = {}

    def latest_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        return {symbol: self.prices[symbol] for symbol in symbols if symbol in self.prices}

    def splits(self, symbols: Sequence[str], start: date, end: date) -> dict[str, str]:
        return {
            symbol: description
            for (symbol, ex_date), description in self.announced_splits.items()
            if symbol in symbols and start <= ex_date <= end
        }


class FakeProvider:
    name = "fake"

    def __init__(self, bars: Mapping[str, pl.DataFrame], through: date) -> None:
        self.bars = dict(bars)
        self.through = through

    def fetch_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        return self.bars[symbol].filter(pl.col("date").is_between(start, min(end, self.through)))


class FakeSharadar:
    def __init__(self, tables: dict[str, list[dict[str, Any]]]) -> None:
        self.tables = tables
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        rows = self.tables[request.url.path.rsplit("/", 1)[-1]]
        if "ticker" in request.url.params:
            tickers = request.url.params["ticker"].split(",")
            rows = [row for row in rows if row.get("ticker", tickers[0]) in tickers]
        skip, limit = int(request.url.params["skip"]), int(request.url.params["limit"])
        page = rows[skip : skip + limit]
        return httpx.Response(200, json={"count": len(page), "data": page})

    def provider(self, page_size: int = 10_000) -> SharadarProvider:
        return SharadarProvider(KEY, transport=httpx.MockTransport(self), page_size=page_size)
