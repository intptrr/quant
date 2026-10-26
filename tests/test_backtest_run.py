from datetime import date

import polars as pl
import pytest
from factories import flat_rates, make_prices, random_walk, trading_with

from quant.backtest.run import run_backtest, trading_days
from quant.core.trading import BrokerTerms

FREE = trading_with()
NO_RATES = flat_rates(0.0)


def constant_targets(prices: pl.DataFrame, weights: dict[str, float]) -> pl.DataFrame:
    return prices.select(
        "date",
        "symbol",
        pl.col("symbol").replace_strict(weights, return_dtype=pl.Float64).alias("weight"),
        pl.lit(0.0).alias("band"),
    )


def test_flat_targets_produce_flat_equity() -> None:
    prices = make_prices({"A": random_walk(1, 30), "B": random_walk(2, 30)})

    result = run_backtest(prices, constant_targets(prices, {"A": 0.0, "B": 0.0}), FREE, NO_RATES)

    assert result.daily["equity"].to_list() == pytest.approx([1.0] * 30)
    assert result.daily["gross_leverage"].max() == 0.0


def test_daily_results_track_leverage_turnover_and_positions() -> None:
    prices = make_prices({"A": random_walk(1, 30), "B": random_walk(2, 30)})

    result = run_backtest(prices, constant_targets(prices, {"A": 0.6, "B": -0.4}), FREE, NO_RATES)

    later = result.daily.tail(10)
    assert later["gross_leverage"].to_list() == pytest.approx([1.0] * 10, abs=0.05)
    assert later["net_leverage"].to_list() == pytest.approx([0.2] * 10, abs=0.05)
    assert result.daily["turnover"][1] == pytest.approx(1.0, abs=0.05)
    assert result.daily["orders"][1] == 2
    assert result.positions.columns == ["date", "symbol", "weight", "traded"]
    assert result.positions.height == 60


def test_returns_compound_to_equity() -> None:
    prices = make_prices({"A": random_walk(1, 30), "B": random_walk(2, 30)})

    daily = run_backtest(
        prices, constant_targets(prices, {"A": 0.5, "B": 0.5}), FREE, NO_RATES
    ).daily

    compounded = (daily["return"] + 1.0).cum_prod()
    assert compounded.to_list() == pytest.approx(daily["equity"].to_list())


def test_slippage_is_a_flat_rate_on_traded_notional() -> None:
    prices = make_prices({"A": [100.0] * 5})

    daily = run_backtest(
        prices, constant_targets(prices, {"A": 0.5}), trading_with(slippage_bps=3.0), NO_RATES
    ).daily

    assert daily["slippage"][1] == pytest.approx(0.5 * 0.0003)


def test_borrowed_cash_pays_the_benchmark_rate_plus_the_margin_spread() -> None:
    prices = make_prices({"A": [100.0] * 5})
    targets = constant_targets(prices, {"A": 2.0})
    rates = pl.DataFrame({"date": [date(2024, 1, 2), date(2024, 1, 4)], "rate": [0.0, 0.126]})

    daily = run_backtest(
        prices, targets, trading_with(BrokerTerms(margin_spread=0.126)), rates
    ).daily

    assert daily["date"][2] == date(2024, 1, 4)
    assert daily["cost"].head(3).to_list() == pytest.approx([0.0, 0.0005, 0.001], abs=1e-6)


def test_commissions_and_interest_are_reported_daily() -> None:
    prices = make_prices({"A": [100.0] * 5})
    terms = BrokerTerms(commission_minimum=1.0, cash_spread=0.0, cash_threshold=10_000.0)

    daily = run_backtest(
        prices, constant_targets(prices, {"A": 0.5}), trading_with(terms), flat_rates(0.252)
    ).daily

    assert daily["commission"][1] == pytest.approx(1.0 / 100_000)
    assert daily["interest"][1] == pytest.approx(0.252 * 0.4 / 252, rel=1e-3)


def test_trading_days_start_with_the_first_position() -> None:
    prices = make_prices({"A": random_walk(1, 10)})

    daily = run_backtest(prices, constant_targets(prices, {"A": 1.0}), FREE, NO_RATES).daily

    assert trading_days(daily)["date"].to_list() == daily["date"].to_list()[1:]
