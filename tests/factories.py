from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import polars as pl

from quant.core.calendar import sessions
from quant.core.trading import BrokerName, BrokerTerms, TradingConfig
from quant.data.adjust import adjust
from quant.data.schema import conform


def make_bars(
    closes: Sequence[float],
    *,
    start: date = date(2024, 1, 2),
    dividends: Mapping[int, float] | None = None,
    splits: Mapping[int, float] | None = None,
) -> pl.DataFrame:
    days = sessions(start, date(start.year + 10, 12, 31))[: len(closes)]
    dividend = [(dividends or {}).get(row, 0.0) for row in range(len(closes))]
    split_factor = [(splits or {}).get(row, 1.0) for row in range(len(closes))]
    return conform(
        pl.DataFrame(
            {
                "date": days,
                "open": closes,
                "high": [close * 1.01 for close in closes],
                "low": [close * 0.99 for close in closes],
                "close": closes,
                "volume": [1_000_000.0] * len(closes),
                "dividend": dividend,
                "split_factor": split_factor,
            }
        )
    )


def set_value(bars: pl.DataFrame, row: int, column: str, value: float | None) -> pl.DataFrame:
    return (
        bars.with_row_index("row")
        .with_columns(
            pl.when(pl.col("row") == row)
            .then(pl.lit(value, dtype=pl.Float64))
            .otherwise(pl.col(column))
            .alias(column)
        )
        .drop("row")
    )


def random_walk(
    seed: int, days: int, *, drift: float = 0.0, volatility: float = 0.01
) -> list[float]:
    returns = np.random.default_rng(seed).normal(drift, volatility, days)
    return (100.0 * np.cumprod(1.0 + returns)).tolist()


def with_symbol(bars: pl.DataFrame, symbol: str) -> pl.DataFrame:
    return (
        adjust(bars)
        .with_columns(pl.lit(symbol).alias("symbol"))
        .select("date", "symbol", pl.exclude("date", "symbol"))
    )


def make_prices(closes: Mapping[str, Sequence[float]]) -> pl.DataFrame:
    return pl.concat(with_symbol(make_bars(series), symbol) for symbol, series in closes.items())


def flat_rates(rate: float = 0.0, *, start: date = date(1990, 1, 2)) -> pl.DataFrame:
    return pl.DataFrame({"date": [start], "rate": [rate]})


def trading_with(
    terms: BrokerTerms | None = None,
    *,
    slippage_bps: float = 0.0,
    min_order_value: float = 0.0,
    max_order_fraction: float = 0.75,
) -> TradingConfig:
    return TradingConfig(
        min_order_value=min_order_value,
        max_order_fraction=max_order_fraction,
        slippage_bps=slippage_bps,
        brokers={BrokerName.IBKR: terms or BrokerTerms()},
    )
