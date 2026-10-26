from datetime import date
from pathlib import Path

import polars as pl
import pytest
from factories import make_bars, make_prices, with_symbol

from quant.strategy import FixedWeights, load_strategy

PROJECT = Path(__file__).parents[1]


def test_the_project_portfolio_is_spy_buy_and_hold() -> None:
    strategy = load_strategy(PROJECT / "configs" / "portfolio.toml")

    assert strategy.symbols == ["SPY"]
    assert strategy.weights == {"SPY": 1.0}


def test_targets_hold_the_weights_every_day_with_a_band_proportional_to_each() -> None:
    prices = make_prices({"A": [100.0] * 3, "B": [50.0] * 3})

    targets = FixedWeights(weights={"A": 0.6, "B": -0.4}, rebalance_band=0.1).targets(prices)

    assert targets.columns == ["date", "symbol", "weight", "band"]
    assert targets["symbol"].to_list() == ["A", "B"] * 3
    assert targets["weight"].to_list() == [0.6, -0.4] * 3
    assert targets["band"].to_list() == pytest.approx([0.06, 0.04] * 3)


def test_the_portfolio_starts_once_every_symbol_has_a_price() -> None:
    prices = pl.concat(
        [
            with_symbol(make_bars([100.0] * 3), "A"),
            with_symbol(make_bars([10.0] * 2, start=date(2024, 1, 3)), "B"),
            with_symbol(make_bars([400.0] * 3), "SPY"),
        ]
    )

    targets = FixedWeights(weights={"A": 0.5, "B": 0.5}).targets(prices)

    assert targets.select("date", "symbol").rows() == [
        (date(2024, 1, 3), "A"),
        (date(2024, 1, 3), "B"),
        (date(2024, 1, 4), "A"),
        (date(2024, 1, 4), "B"),
    ]


def test_a_strategy_needs_weights_and_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="too_short"):
        FixedWeights(weights={})
    with pytest.raises(ValueError, match="extra_forbidden"):
        FixedWeights.model_validate({"weights": {"SPY": 1.0}, "buffer": 0.1})
