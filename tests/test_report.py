from datetime import date

import polars as pl
import pytest
from factories import make_bars, with_symbol

from quant.analytics.report import (
    benchmark_returns,
    common_period,
    format_period,
    split_at,
    yearly_returns,
)


def daily(start: int, end: int) -> pl.DataFrame:
    days = [date(2024, 1, day) for day in range(start, end + 1)]
    return pl.DataFrame({"date": days, "return": [0.001 * day for day in range(len(days))]})


def test_the_benchmark_is_spy_with_dividends_reinvested() -> None:
    prices = pl.concat(
        [
            with_symbol(make_bars([100.0, 101.0, 99.99], dividends={2: 1.01}), "SPY"),
            with_symbol(make_bars([50.0, 60.0, 70.0]), "QQQ"),
        ]
    )

    benchmark = benchmark_returns(prices)

    assert benchmark["return"].to_list() == pytest.approx([0.01, 0.0])


def test_common_period_keeps_only_overlapping_dates() -> None:
    aligned = common_period({"a": daily(2, 20), "b": daily(5, 25)})

    assert {name: (frame["date"][0], frame["date"][-1]) for name, frame in aligned.items()} == {
        "a": (date(2024, 1, 5), date(2024, 1, 20)),
        "b": (date(2024, 1, 5), date(2024, 1, 20)),
    }


def test_splitting_puts_the_split_date_out_of_sample() -> None:
    before, after = split_at({"a": daily(2, 20)}, date(2024, 1, 10))

    assert before["a"]["date"][-1] == date(2024, 1, 9)
    assert after["a"]["date"][0] == date(2024, 1, 10)


def test_yearly_returns_compound_within_each_year() -> None:
    returns = pl.DataFrame(
        {"date": [date(2023, 12, 29), date(2024, 1, 2), date(2024, 1, 3)], "return": [0.1] * 3}
    )

    yearly = yearly_returns({"a": returns})

    assert yearly.rows() == [(2023, pytest.approx(0.1)), (2024, pytest.approx(0.21))]


def test_a_period_has_a_title_line_and_one_aligned_row_per_portfolio() -> None:
    text = format_period("Full period", {"portfolio": daily(2, 9), "SPY buy & hold": daily(2, 9)})

    title, header, first, second = text.splitlines()
    assert title == "Full period: 2024-01-02 to 2024-01-09"
    assert header.startswith("Portfolio") and header.endswith("MaxDD")
    assert first.startswith("portfolio ")
    assert second.startswith("SPY buy & hold")
    assert len({len(header), len(first), len(second)}) == 1
