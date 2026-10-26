import polars as pl
import pytest
from factories import make_bars

from quant.data.adjust import adjust


def test_dividend_is_reinvested_in_total_return() -> None:
    adjusted = adjust(make_bars([100.0, 99.0], dividends={1: 1.0}))

    assert adjusted["total_return"].to_list() == [None, pytest.approx(0.0)]
    assert adjusted["adj_close"].to_list() == pytest.approx([99.0, 99.0])


def test_split_does_not_change_total_return() -> None:
    adjusted = adjust(make_bars([100.0, 50.0], splits={1: 2.0}))

    assert adjusted["total_return"].to_list() == [None, pytest.approx(0.0)]
    assert adjusted["adj_close"].to_list() == pytest.approx([50.0, 50.0])


def test_reverse_split_matches_uso_april_2020() -> None:
    adjusted = adjust(make_bars([2.13, 18.0], splits={1: 0.125}))

    assert adjusted["total_return"][1] == pytest.approx(18.0 / 17.04 - 1.0)


def test_adjusted_prices_are_anchored_to_the_latest_close() -> None:
    adjusted = adjust(make_bars([100.0, 50.0, 51.0, 49.0], dividends={2: 0.5}, splits={1: 2.0}))

    assert adjusted["adj_close"][-1] == adjusted["close"][-1]
    factors = adjusted.select(
        (pl.col(f"adj_{name}") / pl.col(name)).alias(name) for name in ("open", "high", "low")
    )
    expected = (adjusted["adj_close"] / adjusted["close"]).to_list()
    for name in ("open", "high", "low"):
        assert factors[name].to_list() == pytest.approx(expected)


def test_new_dividend_rescales_history_by_a_constant() -> None:
    history = make_bars([100.0, 101.0, 102.0, 103.0])
    extended = make_bars([100.0, 101.0, 102.0, 103.0, 102.5], dividends={4: 1.0})

    ratios = adjust(extended)["adj_close"].head(4) / adjust(history)["adj_close"]

    assert ratios.to_list() == pytest.approx([ratios[0]] * 4, rel=1e-12)
    assert ratios[0] != pytest.approx(1.0)
