from datetime import date
from pathlib import Path

import polars as pl
import pytest
from factories import make_bars, set_value

from quant.data.store import BarStore
from quant.data.sync import HISTORY_START, OVERLAP_ROWS, sync_symbol

CLOSES = [100.0 + day for day in range(30)]


class FakeProvider:
    name = "fake"

    def __init__(self, bars: pl.DataFrame) -> None:
        self.bars = bars
        self.requests: list[tuple[date, date]] = []

    def fetch_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        self.requests.append((start, end))
        return self.bars.filter(pl.col("date").is_between(start, end))


def last_date(bars: pl.DataFrame) -> date:
    return bars["date"][-1]


def stored(store: BarStore) -> pl.DataFrame:
    bars = store.read("fake", "SPY")
    assert bars is not None
    return bars


def test_first_sync_downloads_the_full_history(tmp_path: Path) -> None:
    store, provider = BarStore(tmp_path), FakeProvider(make_bars(CLOSES))

    result = sync_symbol(store, provider, "SPY", last_date(provider.bars))

    assert (result.rows, result.added, result.revised, result.written) == (30, 30, 0, True)
    assert provider.requests[0][0] == HISTORY_START
    assert stored(store).equals(provider.bars)


def test_resync_without_new_data_changes_nothing(tmp_path: Path) -> None:
    store, provider = BarStore(tmp_path), FakeProvider(make_bars(CLOSES))
    end = last_date(provider.bars)
    sync_symbol(store, provider, "SPY", end)

    result = sync_symbol(store, provider, "SPY", end)

    assert (result.added, result.revised, result.written) == (0, 0, False)
    assert provider.requests[1][0] == provider.bars["date"][-OVERLAP_ROWS]


def test_incremental_sync_appends_new_bars(tmp_path: Path) -> None:
    full = make_bars(CLOSES)
    store, provider = BarStore(tmp_path), FakeProvider(full.head(25))
    sync_symbol(store, provider, "SPY", last_date(provider.bars))
    provider.bars = full

    result = sync_symbol(store, provider, "SPY", last_date(full))

    assert (result.rows, result.added, result.revised) == (30, 5, 0)
    assert stored(store).equals(full)


def test_revisions_inside_the_overlap_are_counted_and_applied(tmp_path: Path) -> None:
    store, provider = BarStore(tmp_path), FakeProvider(make_bars(CLOSES))
    end = last_date(provider.bars)
    sync_symbol(store, provider, "SPY", end)
    provider.bars = set_value(provider.bars, 28, "close", 150.0)

    result = sync_symbol(store, provider, "SPY", end)

    assert (result.revised, result.written) == (1, True)
    assert stored(store)["close"][28] == 150.0


def test_full_sync_redownloads_from_the_history_start(tmp_path: Path) -> None:
    store, provider = BarStore(tmp_path), FakeProvider(make_bars(CLOSES))
    end = last_date(provider.bars)
    sync_symbol(store, provider, "SPY", end)

    result = sync_symbol(store, provider, "SPY", end, full=True)

    assert provider.requests[1][0] == HISTORY_START
    assert (result.revised, result.written) == (0, False)


def test_sync_fails_when_the_provider_returns_no_bars(tmp_path: Path) -> None:
    provider = FakeProvider(make_bars(CLOSES).head(0))

    with pytest.raises(ValueError, match="fake returned no bars for SPY"):
        sync_symbol(BarStore(tmp_path), provider, "SPY", date(2024, 3, 1))
