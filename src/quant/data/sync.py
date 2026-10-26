from dataclasses import dataclass
from datetime import date

import polars as pl

from quant.data.providers import PriceProvider
from quant.data.schema import BAR_SCHEMA, conform, empty_bars
from quant.data.store import BarStore

HISTORY_START = date(1990, 1, 1)
OVERLAP_ROWS = 10
REVISION_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class SyncResult:
    symbol: str
    rows: int
    added: int
    revised: int
    written: bool


def sync_symbol(
    store: BarStore, provider: PriceProvider, symbol: str, end: date, *, full: bool = False
) -> SyncResult:
    existing = store.read(provider.name, symbol)
    previous = empty_bars() if existing is None else existing
    start = HISTORY_START if full or previous.is_empty() else overlap_start(previous)
    incoming = conform(provider.fetch_daily(symbol, start, end))
    if incoming.is_empty():
        raise ValueError(f"{provider.name} returned no bars for {symbol} between {start} and {end}")
    merged, revised = merge_bars(previous, incoming)
    written = existing is None or not merged.equals(existing)
    if written:
        store.write(provider.name, symbol, merged)
    return SyncResult(symbol, merged.height, merged.height - previous.height, revised, written)


def overlap_start(bars: pl.DataFrame) -> date:
    return bars["date"][-min(OVERLAP_ROWS, bars.height)]


def merge_bars(existing: pl.DataFrame, incoming: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    start = incoming["date"].min()
    kept = existing.filter(pl.col("date") < start)
    replaced = existing.filter(pl.col("date") >= start)
    merged = pl.concat([kept, incoming]).sort("date")
    return merged, count_revisions(replaced, incoming)


def count_revisions(previous: pl.DataFrame, current: pl.DataFrame) -> int:
    joined = previous.join(current, on="date", how="left", suffix="_new")
    changed = pl.any_horizontal(
        pl.col(f"{name}_new").is_null()
        | (
            (pl.col(name) - pl.col(f"{name}_new")).abs()
            > REVISION_TOLERANCE * pl.col(name).abs().clip(lower_bound=1.0)
        )
        for name in BAR_SCHEMA
        if name != "date"
    )
    return joined.filter(changed).height
