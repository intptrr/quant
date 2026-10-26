from collections.abc import Mapping
from dataclasses import asdict
from datetime import date

import polars as pl

from quant.analytics.metrics import TRADING_DAYS, summarize
from quant.data.rates import rates_on

BENCHMARK = "SPY"
METRICS: tuple[tuple[str, str, str], ...] = (
    ("CAGR", "cagr", ".2%"),
    ("Vol", "volatility", ".2%"),
    ("Sharpe", "sharpe", ".2f"),
    ("MaxDD", "max_drawdown", ".2%"),
)


def benchmark_returns(prices: pl.DataFrame) -> pl.DataFrame:
    return (
        prices.filter(pl.col("symbol") == BENCHMARK)
        .select("date", pl.col("total_return").alias("return"))
        .drop_nulls()
    )


def common_period(returns: Mapping[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    start = max(frame.select(pl.col("date").min()).item() for frame in returns.values())
    end = min(frame.select(pl.col("date").max()).item() for frame in returns.values())
    return {
        name: frame.filter(pl.col("date").is_between(start, end)) for name, frame in returns.items()
    }


def split_at(
    returns: Mapping[str, pl.DataFrame], day: date
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    before = pl.col("date") < day
    return (
        {name: frame.filter(before) for name, frame in returns.items()},
        {name: frame.filter(~before) for name, frame in returns.items()},
    )


def yearly_returns(returns: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    yearly = [
        frame.group_by(pl.col("date").dt.year().alias("year"))
        .agg(((pl.col("return") + 1.0).product() - 1.0).alias(name))
        .sort("year")
        for name, frame in returns.items()
    ]
    joined = yearly[0]
    for frame in yearly[1:]:
        joined = joined.join(frame, on="year", how="full", coalesce=True)
    return joined.sort("year")


def format_period(title: str, returns: Mapping[str, pl.DataFrame], rates: pl.DataFrame) -> str:
    dates = next(iter(returns.values()))["date"]
    headers = ["Portfolio", *(label for label, _, _ in METRICS)]
    rows = []
    for name, frame in returns.items():
        cash = rates_on(frame["date"], rates) / TRADING_DAYS
        summary = asdict(summarize(frame["return"], cash))
        rows.append([name, *(format(summary[field], spec) for _, field, spec in METRICS)])
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows, strict=True)]
    table = [
        "  ".join(
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(line, widths, strict=True))
        )
        for line in [headers, *rows]
    ]
    return "\n".join([f"{title}: {dates[0]} to {dates[-1]}", *table])


def format_yearly(yearly: pl.DataFrame) -> str:
    names = [name for name in yearly.columns if name != "year"]
    widths = [max(len(name), 7) for name in names]
    lines = [
        "Year  " + "  ".join(name.rjust(width) for name, width in zip(names, widths, strict=True))
    ]
    for row in yearly.iter_rows(named=True):
        cells = ["" if row[name] is None else format(row[name], ".1%") for name in names]
        lines.append(
            f"{row['year']}  "
            + "  ".join(cell.rjust(width) for cell, width in zip(cells, widths, strict=True))
        )
    return "\n".join(lines)


def format_profiles(dailies: Mapping[str, pl.DataFrame], benchmark: pl.DataFrame) -> str:
    width = max(len(name) for name in dailies)
    return "\n".join(
        f"{name.ljust(width)}  {format_profile(daily, benchmark)}"
        for name, daily in dailies.items()
    )


def format_profile(daily: pl.DataFrame, benchmark: pl.DataFrame) -> str:
    correlation = (
        daily.select("date", "return")
        .join(benchmark.select("date", pl.col("return").alias("benchmark")), on="date")
        .select(pl.corr("return", "benchmark"))
        .item()
    )
    profile = daily.select(
        pl.col("gross_leverage").mean(),
        pl.col("net_leverage").mean(),
        pl.col("turnover").mean() * TRADING_DAYS,
        pl.col("cost").mean() * TRADING_DAYS,
    ).row(0)
    gross, net, turnover, cost = profile
    return (
        f"avg gross {gross:.2f}x, avg net {net:+.2f}x, turnover {turnover:.1f}x/yr, "
        f"costs {cost:.2%}/yr, correlation to {BENCHMARK} {correlation:+.2f}"
    )
