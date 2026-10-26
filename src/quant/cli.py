import logging
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from quant.analytics.report import (
    BENCHMARK,
    benchmark_returns,
    common_period,
    format_period,
    format_profile,
    format_yearly,
    split_at,
    yearly_returns,
)
from quant.backtest.run import run_backtest, trading_days
from quant.core.calendar import last_closed_session, new_york_today
from quant.core.trading import BrokerName, TradingConfig, load_trading_config
from quant.data.prices import load_adjusted
from quant.data.providers import SHARADAR, sharadar_provider
from quant.data.quality import Issue, Severity, check_bars
from quant.data.rates import RateStore, fetch_benchmark_rates
from quant.data.store import BarStore
from quant.data.sync import sync_symbol
from quant.execution.alpaca import AlpacaBroker, AlpacaMarketWatch
from quant.execution.broker import Broker
from quant.execution.ibkr import IbkrBroker
from quant.live.jobs import (
    Desk,
    JobReport,
    broker_check,
    initialize,
    post_open,
    pre_open,
    tracking_report,
    upcoming_session,
)
from quant.live.state import StateStore
from quant.settings import Settings
from quant.strategy import load_strategy

DEFAULT_STRATEGY = Path("configs/portfolio.toml")
DEFAULT_TRADING = Path("configs/trading.toml")
BENCHMARK_LABEL = f"{BENCHMARK} buy & hold"
STATE_DIR = Path("state")

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
data_app = typer.Typer(no_args_is_help=True, help="Sync and check daily bars.")
paper_app = typer.Typer(
    no_args_is_help=True, help="Paper trading at the opening auction (IBKR or Alpaca)."
)
app.add_typer(data_app, name="data")
app.add_typer(paper_app, name="paper")

StrategyOption = Annotated[
    Path, typer.Option("--strategy", "-s", help="Strategy config: target weights and band.")
]
TradingOption = Annotated[
    Path, typer.Option("--trading", "-t", help="Trading config: broker, capital, costs, limits.")
]
BrokerOption = Annotated[
    BrokerName | None,
    typer.Option("--broker", "-b", help="Broker (default: the one in the trading config)."),
]
SymbolsOption = Annotated[
    list[str] | None, typer.Option("--symbol", help="Only these symbols (repeatable).")
]
DATE_FORMATS = ["%Y-%m-%d"]


@data_app.command("sync")
def data_sync(
    strategy: StrategyOption = DEFAULT_STRATEGY,
    symbols: SymbolsOption = None,
    full: Annotated[bool, typer.Option(help="Re-download the full history.")] = False,
) -> None:
    """Download daily bars and the Fed funds rate, then run quality checks."""
    settings = Settings()
    store = BarStore(settings.data_dir)
    provider = sharadar_provider(settings)
    now = datetime.now(UTC)
    issues: list[Issue] = []
    for symbol in _symbols(strategy, symbols):
        result = sync_symbol(store, provider, symbol, new_york_today(now), full=full)
        typer.echo(
            f"{symbol:<6} rows={result.rows:<6} added={result.added:<6} "
            f"revised={result.revised:<4} written={result.written}"
        )
        issues.extend(check_bars(symbol, store.read(SHARADAR, symbol), last_closed_session(now)))
    rates = fetch_benchmark_rates()
    RateStore(settings.data_dir).write(rates)
    typer.echo(f"Fed funds rows={rates.height} last={rates['date'][-1]}")
    _report(issues)


@data_app.command("check")
def data_check(strategy: StrategyOption = DEFAULT_STRATEGY, symbols: SymbolsOption = None) -> None:
    """Run quality checks on stored bars."""
    store = BarStore(Settings().data_dir)
    expected_last = last_closed_session(datetime.now(UTC))
    _report(
        [
            issue
            for symbol in _symbols(strategy, symbols)
            for issue in check_bars(symbol, store.read(SHARADAR, symbol), expected_last)
        ]
    )


@app.command("backtest")
def backtest(
    strategy: StrategyOption = DEFAULT_STRATEGY,
    trading: TradingOption = DEFAULT_TRADING,
    split: Annotated[
        datetime | None,
        typer.Option(
            formats=DATE_FORMATS,
            help="Also report before (in sample) and from (out of sample) this date.",
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Write daily and position parquet files here.")
    ] = None,
) -> None:
    """Backtest the strategy against SPY buy-and-hold."""
    settings = Settings()
    portfolio = load_strategy(strategy)
    prices = load_adjusted(
        BarStore(settings.data_dir), SHARADAR, _with_benchmark(portfolio.symbols)
    )
    rates = RateStore(settings.data_dir).read()
    result = run_backtest(prices, portfolio.targets(prices), load_trading_config(trading), rates)
    daily = trading_days(result.daily)
    portfolios = common_period(
        {
            strategy.stem: daily.select("date", "return"),
            BENCHMARK_LABEL: benchmark_returns(prices),
        }
    )
    sections = [format_period("Full period", portfolios, rates)]
    if split is not None:
        before, after = split_at(portfolios, split.date())
        if any(frame.is_empty() for frame in (*before.values(), *after.values())):
            raise typer.BadParameter("must fall inside the backtest period", param_hint="--split")
        sections += [
            format_period("In sample", before, rates),
            format_period("Out of sample", after, rates),
        ]
    sections += [
        format_profile(daily, portfolios[BENCHMARK_LABEL]),
        format_yearly(yearly_returns(portfolios)),
    ]
    typer.echo("\n\n".join(sections))
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        result.daily.write_parquet(output / "daily.parquet")
        result.positions.write_parquet(output / "positions.parquet")


@paper_app.command("check")
def paper_check(
    broker: BrokerOption = None,
    strategy: StrategyOption = DEFAULT_STRATEGY,
    trading: TradingOption = DEFAULT_TRADING,
) -> None:
    """Check the broker connection, account, short availability and market data."""
    _run_job(broker_check, broker, strategy, trading)


@paper_app.command("init")
def paper_init(
    capital: Annotated[
        float | None, typer.Option(help="Starting capital (default: account equity).")
    ] = None,
    broker: BrokerOption = None,
    strategy: StrategyOption = DEFAULT_STRATEGY,
    trading: TradingOption = DEFAULT_TRADING,
) -> None:
    """Start the shadow ledger; the paper account must have no positions."""
    now = datetime.now(UTC)
    _run_job(lambda desk: initialize(desk, now, capital), broker, strategy, trading)


@paper_app.command("pre-open")
def paper_pre_open(
    session: Annotated[
        datetime | None,
        typer.Option(
            formats=DATE_FORMATS,
            help="Trading session (default: the next opening auction still ahead).",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Plan orders without submitting them.")] = False,
    broker: BrokerOption = None,
    strategy: StrategyOption = DEFAULT_STRATEGY,
    trading: TradingOption = DEFAULT_TRADING,
) -> None:
    """Sync data, close the ledger, then queue orders for the opening auction."""
    now = datetime.now(UTC)
    day = upcoming_session(now) if session is None else session.date()
    _run_job(lambda desk: pre_open(desk, day, now, submit=not dry_run), broker, strategy, trading)


@paper_app.command("post-open")
def paper_post_open(
    session: Annotated[
        datetime | None,
        typer.Option(formats=DATE_FORMATS, help="Trading session (default: today in New York)."),
    ] = None,
    broker: BrokerOption = None,
    strategy: StrategyOption = DEFAULT_STRATEGY,
    trading: TradingOption = DEFAULT_TRADING,
) -> None:
    """Record opening fills and reconcile positions with the broker."""
    day = new_york_today(datetime.now(UTC)) if session is None else session.date()
    _run_job(lambda desk: post_open(desk, day), broker, strategy, trading)


@paper_app.command("report")
def paper_report(
    broker: BrokerOption = None,
    strategy: StrategyOption = DEFAULT_STRATEGY,
    trading: TradingOption = DEFAULT_TRADING,
) -> None:
    """Compare the paper ledger with a backtest replay over the same days."""
    _run_job(tracking_report, broker, strategy, trading)


def _run_job(
    job: Callable[[Desk], JobReport], broker: BrokerName | None, strategy: Path, trading: Path
) -> None:
    setup = load_trading_config(trading)
    desk = _desk(Settings(), setup if broker is None else setup.using(broker), strategy)
    with closing(desk.broker):
        report = job(desk)
    typer.echo(report.text)
    if not report.ok:
        raise typer.Exit(code=1)


def _desk(settings: Settings, trading: TradingConfig, strategy: Path) -> Desk:
    key, secret = _alpaca_keys(settings)
    return Desk(
        bars=BarStore(settings.data_dir),
        rates=RateStore(settings.data_dir),
        provider=sharadar_provider(settings),
        broker=_broker(settings, trading.broker),
        market=AlpacaMarketWatch(key, secret),
        state=StateStore(settings.data_dir / STATE_DIR / f"paper-{trading.broker}.sqlite3"),
        strategy=load_strategy(strategy),
        trading=trading,
    )


def _broker(settings: Settings, name: BrokerName) -> Broker:
    match name:
        case BrokerName.ALPACA:
            return AlpacaBroker(*_alpaca_keys(settings))
        case BrokerName.IBKR:
            if settings.ibkr_account is None:
                raise ValueError("IBKR_ACCOUNT is not set; add the account ID to .env")
            return IbkrBroker(
                settings.ibkr_host,
                settings.ibkr_port,
                settings.ibkr_client_id,
                settings.ibkr_account,
            )


def _alpaca_keys(settings: Settings) -> tuple[str, str]:
    if settings.alpaca_api_key is None or settings.alpaca_secret_key is None:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are not set; add them to .env")
    return settings.alpaca_api_key.get_secret_value(), settings.alpaca_secret_key.get_secret_value()


def _symbols(strategy: Path, symbols: list[str] | None) -> list[str]:
    if symbols:
        return [symbol.upper() for symbol in symbols]
    return _with_benchmark(load_strategy(strategy).symbols)


def _with_benchmark(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys([*symbols, BENCHMARK]))


def _report(issues: list[Issue]) -> None:
    for issue in issues:
        typer.echo(f"{issue.severity:<7} {issue.symbol:<6} {issue.check:<24} {issue.detail}")
    errors = sum(issue.severity is Severity.ERROR for issue in issues)
    typer.echo(f"{errors} errors, {len(issues) - errors} warnings")
    if errors:
        raise typer.Exit(code=1)


def main() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app()
