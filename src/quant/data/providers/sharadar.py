from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx
import polars as pl

from quant.data.quality import describe_dates
from quant.data.schema import conform, empty_bars

BASE_URL = "https://api.sharadar.com/v1.0/data"
PAGE_SIZE = 10_000
PRICE_TABLES = ("funds", "stocks")
PRICE_FIELDS = "date,open,high,low,close,volume,closeunadj"
ACTION_FIELDS = "date,action,value"
LISTING_FIELDS = "ticker,table,firstpricedate"
MAX_START_GAP = timedelta(days=7)
SPLIT_TOLERANCE = 1e-3
EVENT_SCHEMA = pl.Schema(
    {"date": pl.Date(), "dividend": pl.Float64(), "split_factor": pl.Float64()}
)


class SharadarError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Listing:
    table: str
    first_price_date: date


class SharadarProvider:
    name = "sharadar"

    def __init__(
        self,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self._api_key = api_key
        self._page_size = page_size
        self._client = httpx.Client(base_url=BASE_URL, timeout=60.0, transport=transport)

    def fetch_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        listing = self._listing(symbol)
        window = {"ticker": symbol, "from": start.isoformat(), "to": end.isoformat()}
        prices = self._records(
            listing.table, {**window, "fields": PRICE_FIELDS, "sort": "date.asc"}
        )
        actions = self._records("actions", {**window, "fields": ACTION_FIELDS, "sort": "date.asc"})
        bars = to_bars(symbol, prices, actions)
        expected_start = max(start, listing.first_price_date)
        if not bars.is_empty() and bars["date"][0] > expected_start + MAX_START_GAP:
            raise SharadarError(
                f"{symbol}: prices start {bars['date'][0]} but Sharadar has history from "
                f"{listing.first_price_date}; check SHARADAR_API_KEY and your subscription"
            )
        return bars

    def _listing(self, symbol: str) -> Listing:
        rows = [
            row
            for row in self._records("tickers", {"ticker": symbol, "fields": LISTING_FIELDS})
            if row.get("table") in PRICE_TABLES and row.get("firstpricedate")
        ]
        if len(rows) != 1:
            raise SharadarError(
                f"{symbol} has no price history in Sharadar's funds or stocks tables"
            )
        return Listing(rows[0]["table"], date.fromisoformat(rows[0]["firstpricedate"]))

    def _records(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        while True:
            response = self._client.get(
                f"/{table}",
                params={
                    **params,
                    "api_key": self._api_key,
                    "format": "json",
                    "limit": self._page_size,
                    "skip": len(records),
                },
            )
            if response.is_error:
                subject = params.get("ticker", "all tickers")
                detail = response.text[:300].replace(self._api_key, "***")
                hint = (
                    "; check SHARADAR_API_KEY and your subscription"
                    if response.status_code in (401, 403)
                    else ""
                )
                raise SharadarError(
                    f"{table} request for {subject} failed with {response.status_code}: "
                    f"{detail}{hint}"
                )
            page = response.json()["data"]
            records.extend(page)
            if len(page) < self._page_size:
                return records


def to_bars(
    symbol: str, prices: list[dict[str, Any]], actions: list[dict[str, Any]]
) -> pl.DataFrame:
    if not prices:
        return empty_bars()
    frame = pl.DataFrame(prices, infer_schema_length=None).with_columns(
        pl.col("date").str.to_date()
    )
    events = corporate_events(actions)
    if orphans := events.join(frame, on="date", how="anti")["date"].sort().to_list():
        raise SharadarError(
            f"{symbol}: corporate actions without prices on {describe_dates(orphans)}"
        )
    unadjustment = pl.col("closeunadj") / pl.col("close")
    bars = frame.join(events, on="date", how="left", maintain_order="left").select(
        "date",
        *[(pl.col(name) * unadjustment).alias(name) for name in ("open", "high", "low")],
        pl.col("closeunadj").alias("close"),
        (pl.col("volume") / unadjustment).alias("volume"),
        (pl.col("dividend") * unadjustment).fill_null(0.0).alias("dividend"),
        pl.col("split_factor").fill_null(1.0),
        unadjustment.alias("unadjustment"),
    )
    implied_split = pl.col("unadjustment").shift(1) / pl.col("unadjustment")
    if mismatched := bars.filter(
        (implied_split / pl.col("split_factor") - 1.0).abs() > SPLIT_TOLERANCE
    )["date"].to_list():
        raise SharadarError(
            f"{symbol}: split-adjusted prices disagree with split actions on "
            f"{describe_dates(mismatched)}"
        )
    return conform(bars)


def corporate_events(actions: list[dict[str, Any]]) -> pl.DataFrame:
    relevant = [action for action in actions if action.get("action") in ("dividend", "split")]
    if not relevant:
        return pl.DataFrame(schema=EVENT_SCHEMA)
    frame = pl.DataFrame(relevant, infer_schema_length=None).with_columns(
        pl.col("date").str.to_date(), pl.col("value").cast(pl.Float64)
    )
    return frame.group_by("date").agg(
        pl.col("value").filter(pl.col("action") == "dividend").sum().alias("dividend"),
        pl.col("value").filter(pl.col("action") == "split").product().alias("split_factor"),
    )
