from dataclasses import dataclass

import numpy as np
import polars as pl

from quant.backtest.engine import Costs, Orders, simulate
from quant.core.trading import TradingConfig
from quant.data.rates import rates_on

BASIS_POINT = 1e-4


@dataclass(frozen=True, slots=True)
class BacktestResult:
    daily: pl.DataFrame
    positions: pl.DataFrame


def run_backtest(
    prices: pl.DataFrame, targets: pl.DataFrame, trading: TradingConfig, rates: pl.DataFrame
) -> BacktestResult:
    symbols = sorted(targets["symbol"].unique().to_list())
    frame = (
        prices.select("date", "symbol", "open", "adj_open", "adj_close")
        .filter(pl.col("symbol").is_in(symbols))
        .join(
            targets.select("date", "symbol", "weight", "band"),
            on=["date", "symbol"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            (
                pl.col("adj_open") / pl.col("adj_close").shift(1).over("symbol", order_by="date")
                - 1.0
            ).alias("overnight"),
            (pl.col("adj_close") / pl.col("adj_open") - 1.0).alias("intraday"),
        )
    )
    dates = frame.select(pl.col("date").unique().sort())["date"]

    def wide(column: str) -> np.ndarray:
        return (
            frame.pivot(on="symbol", index="date", values=column)
            .sort("date")
            .select(symbols)
            .to_numpy()
        )

    terms = trading.terms
    benchmark = rates_on(dates, rates)
    simulation = simulate(
        wide("overnight"),
        wide("intraday"),
        wide("weight"),
        wide("band"),
        Costs(
            slippage=trading.slippage_bps * BASIS_POINT,
            borrow_fee=terms.borrow_fee,
            debit_rate=terms.debit_rate(benchmark),
            credit_rate=terms.credit_rate(benchmark, trading.capital),
            credit_threshold=terms.cash_threshold / trading.capital,
        ),
        Orders(wide("open"), trading.capital, terms, trading.min_order_value),
    )
    previous_equity = np.concatenate(([1.0], simulation.equity[:-1]))
    daily = pl.DataFrame(
        {
            "date": dates,
            "equity": simulation.equity,
            "return": simulation.equity / previous_equity - 1.0,
            "gross_leverage": np.abs(simulation.weights).sum(axis=1),
            "net_leverage": simulation.weights.sum(axis=1),
            "turnover": np.abs(simulation.traded).sum(axis=1) / previous_equity,
            "orders": simulation.orders,
            "cost": simulation.costs / previous_equity,
            "slippage": simulation.slippage / previous_equity,
            "commission": simulation.commissions / previous_equity,
            "borrow": simulation.borrow / previous_equity,
            "financing": simulation.financing / previous_equity,
            "interest": simulation.interest / previous_equity,
        }
    )
    rows = np.repeat(np.arange(dates.len()), len(symbols))
    positions = pl.DataFrame(
        {
            "date": dates.gather(rows),
            "symbol": symbols * dates.len(),
            "weight": simulation.weights.reshape(-1),
            "traded": (simulation.traded / previous_equity[:, None]).reshape(-1),
        }
    )
    return BacktestResult(daily, positions)


def trading_days(daily: pl.DataFrame) -> pl.DataFrame:
    return daily.filter(pl.col("gross_leverage").cum_max() > 0.0)
