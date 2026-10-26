import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl

from quant.analytics.metrics import annualized_volatility
from quant.backtest.run import run_backtest
from quant.core.calendar import (
    at_new_york_time,
    is_session,
    last_closed_session,
    new_york_today,
    next_session,
    previous_session,
    sessions,
)
from quant.core.trading import BrokerTerms, TradingConfig
from quant.data.prices import load_adjusted
from quant.data.providers import PriceProvider
from quant.data.quality import Severity, check_bars
from quant.data.rates import RateStore, rates_on
from quant.data.store import BarStore
from quant.data.sync import sync_symbol
from quant.execution.broker import (
    TERMINAL_STATUSES,
    Broker,
    BrokerOrder,
    MarketWatch,
    OrderRejectedError,
)
from quant.live.ledger import Bar, Carry, Fill, LedgerDay, close_day, opening_day
from quant.live.planner import OrderIntent, Target, below_minimum, plan_orders
from quant.live.risk import Rejection, screen_orders
from quant.live.state import OrderRecord, StateStore
from quant.strategy import Strategy

SUBMISSION_OPENS = time(19, 0)
SUBMISSION_CUTOFF = time(9, 28)
POSITION_TOLERANCE = 0.5
BASIS_POINTS = 1e4


@dataclass(frozen=True, slots=True)
class Desk:
    bars: BarStore
    rates: RateStore
    provider: PriceProvider
    broker: Broker
    market: MarketWatch
    state: StateStore
    strategy: Strategy
    trading: TradingConfig


@dataclass(frozen=True, slots=True)
class JobReport:
    title: str
    lines: list[str]
    ok: bool = True

    @property
    def text(self) -> str:
        heading = self.title if self.ok else f"[ATTENTION] {self.title}"
        return "\n".join([heading, *self.lines])


def initialize(desk: Desk, now: datetime, capital: float | None) -> JobReport:
    title = "paper init"
    if desk.state.latest_day() is not None:
        return JobReport(title, ["paper trading is already initialized"], ok=False)
    positions, open_orders = desk.broker.positions(), desk.broker.open_orders()
    if positions or open_orders:
        detail = f"{len(positions)} positions and {len(open_orders)} open orders at the broker"
        return JobReport(title, [f"the broker account must be flat: {detail}"], ok=False)
    starting = desk.broker.equity() if capital is None else capital
    day = opening_day(last_closed_session(now), starting)
    desk.state.save_day(day)
    return JobReport(title, [f"ledger starts {day.date} with {day.cash:,.2f} cash"])


def broker_check(desk: Desk) -> JobReport:
    symbols = desk.strategy.symbols
    positions, open_orders = desk.broker.positions(), desk.broker.open_orders()
    shortable = desk.broker.shortable(symbols)
    prices = desk.market.latest_prices(symbols)
    blocked = sorted(symbol for symbol, allowed in shortable.items() if not allowed)
    missing = sorted(set(symbols) - set(prices))
    lines = [
        f"equity {desk.broker.equity():,.2f}",
        f"{len(positions)} positions, {len(open_orders)} open orders",
        f"cannot short: {', '.join(blocked) or 'none'}",
        f"latest prices for {len(prices)} of {len(symbols)} symbols",
    ]
    if missing:
        lines.append(f"no latest price for {', '.join(missing)}")
    return JobReport(f"{desk.trading.broker} check", lines, ok=not missing)


def upcoming_session(now: datetime) -> date:
    today = new_york_today(now)
    if is_session(today) and now < at_new_york_time(today, SUBMISSION_CUTOFF):
        return today
    return next_session(today)


def pre_open(desk: Desk, session: date, now: datetime, *, submit: bool) -> JobReport:
    title = f"pre-open {session}" + ("" if submit else " (dry run)")
    if not is_session(session):
        return JobReport(title, ["not a trading session"])
    previous = previous_session(session)
    opens = at_new_york_time(previous, SUBMISSION_OPENS)
    closes = at_new_york_time(session, SUBMISSION_CUTOFF)
    if submit and not opens <= now < closes:
        window = f"{opens:%Y-%m-%d %H:%M} to {closes:%Y-%m-%d %H:%M} ET"
        return JobReport(title, [f"outside the submission window {window}"], ok=False)
    if problems := refresh_data(desk, session, previous):
        return JobReport(title, ["market data is not ready:", *problems], ok=False)
    ledger, lines = close_ledger(desk, previous, save=submit)
    if mismatches := reconcile(ledger.positions, desk.broker.positions()):
        return JobReport(
            title, [*lines, "positions differ from the broker:", *mismatches], ok=False
        )
    targets = session_targets(desk, previous)
    closes = session_bars(desk, previous, set(targets) | set(ledger.positions))
    planned = plan_orders(
        targets,
        {symbol: bar.close for symbol, bar in closes.items()},
        ledger.positions,
        ledger.equity,
    )
    minimum = desk.trading.min_order_value
    small = [intent.symbol for intent in planned if below_minimum(intent, minimum)]
    intents = [intent for intent in planned if not below_minimum(intent, minimum)]
    symbols = [intent.symbol for intent in intents]
    approved, rejected = screen_orders(
        intents,
        equity=ledger.equity,
        latest_prices=desk.market.latest_prices(symbols),
        shortable=desk.broker.shortable(
            [intent.symbol for intent in intents if intent.resulting < 0 and intent.adds_risk]
        ),
        splits=desk.market.splits(symbols, previous + timedelta(days=1), session),
        drawdown=ledger.drawdown,
        trading=desk.trading,
    )
    saved = {order.symbol for order in desk.state.orders(session)}
    if submit:
        refused = submit_orders(desk, session, [i for i in approved if i.symbol not in saved])
        refused_symbols = {item.intent.symbol for item in refused}
        approved = [intent for intent in approved if intent.symbol not in refused_symbols]
        rejected += refused
    lines += [describe_intent(intent, intent.symbol in saved) for intent in approved]
    lines += [f"rejected {describe_intent(item.intent)}: {item.reason}" for item in rejected]
    if small:
        lines.append(f"below the ${minimum:,.0f} minimum order, not traded: {', '.join(small)}")
    if not intents:
        lines.append("no orders needed")
    return JobReport(title, lines, ok=not any(item.needs_attention for item in rejected))


def post_open(desk: Desk, session: date) -> JobReport:
    title = f"post-open {session}"
    if not is_session(session):
        return JobReport(title, ["not a trading session"])
    orders = refresh_orders(desk, session)
    if not orders:
        return JobReport(title, ["no orders for this session"])
    lines = [describe_order(order) for order in orders]
    ledger = desk.state.latest_day()
    previous = previous_session(session)
    if ledger is None or ledger.date != previous:
        state = "is not initialized" if ledger is None else f"ends on {ledger.date}"
        return JobReport(title, [*lines, f"the ledger {state}, expected {previous}"], ok=False)
    expected = dict(ledger.positions)
    for order in orders:
        expected[order.symbol] = expected.get(order.symbol, 0.0) + order.filled_quantity
    mismatches = reconcile(expected, desk.broker.positions())
    unfilled = [order for order in orders if abs(order.filled_quantity) < abs(order.quantity)]
    return JobReport(title, [*lines, *mismatches], ok=not unfilled and not mismatches)


def tracking_report(desk: Desk) -> JobReport:
    title = "paper tracking"
    days = desk.state.days()
    if len(days) < 2:
        return JobReport(title, ["no closed trading day yet"])
    start = days[0].date
    live = pl.DataFrame(
        {"date": [day.date for day in days], "equity": [day.equity for day in days]}
    ).select("date", (pl.col("equity") / pl.col("equity").shift(1) - 1.0).alias("live"))
    joined = live.join(replay_returns(desk, start), on="date").drop_nulls()
    growth = joined.select((pl.col("live", "replay") + 1.0).product() - 1.0).row(0)
    lines = [
        f"{joined.height} trading days since {start}",
        f"paper {growth[0]:+.2%}, backtest replay {growth[1]:+.2%}",
    ]
    if joined.height > 1:
        error = annualized_volatility(joined["live"] - joined["replay"])
        correlation = joined.select(pl.corr("live", "replay")).item()
        lines.append(f"tracking error {error:.2%} a year, correlation {correlation:.2f}")
    lines += slippage_lines(desk, [day.date for day in days[1:]])
    return JobReport(title, lines)


def replay_returns(desk: Desk, start: date) -> pl.DataFrame:
    prices = market_prices(desk)
    since = pl.col("date") >= start
    targets = desk.strategy.targets(prices).filter(since)
    replay = run_backtest(prices.filter(since), targets, desk.trading, desk.rates.read())
    return replay.daily.select("date", pl.col("return").alias("replay"))


def market_prices(desk: Desk) -> pl.DataFrame:
    return load_adjusted(desk.bars, desk.provider.name, desk.strategy.symbols)


def slippage_lines(desk: Desk, closed: Sequence[date]) -> list[str]:
    rows: list[tuple[str, float, float, float]] = []
    for session in closed:
        fills = session_fills(desk.state.orders(session), desk.trading.terms)
        bars = session_bars(desk, session, {fill.symbol for fill in fills})
        rows += [
            (
                fill.symbol,
                abs(fill.quantity) * fill.price,
                slippage_bps(fill, bars[fill.symbol].open),
                desk.trading.slippage_bps,
            )
            for fill in fills
        ]
    if not rows:
        return ["no fills yet"]
    fills = pl.DataFrame(rows, schema=["symbol", "notional", "bps", "assumed"], orient="row")
    stats = [
        pl.len().alias("fills"),
        weighted("bps").alias("mean"),
        pl.col("bps").max().alias("worst"),
        weighted("assumed").alias("assumed"),
    ]
    summary = pl.concat(
        [
            fills.group_by("symbol").agg(stats).sort("symbol"),
            fills.select(pl.lit("all").alias("symbol"), *stats),
        ]
    )
    header = f"{'symbol':<7}{'fills':>6}{'mean':>8}{'worst':>8}{'assumed':>9}"
    return [
        "slippage against the open in bps, positive is worse:",
        header,
        *(
            f"{symbol:<7}{count:>6}{mean:>+8.1f}{worst:>+8.1f}{assumed:>9.1f}"
            for symbol, count, mean, worst, assumed in summary.iter_rows()
        ),
    ]


def weighted(column: str) -> pl.Expr:
    return (pl.col(column) * pl.col("notional")).sum() / pl.col("notional").sum()


def refresh_data(desk: Desk, today: date, previous: date) -> list[str]:
    problems: list[str] = []
    for symbol in desk.strategy.symbols:
        sync_symbol(desk.bars, desk.provider, symbol, today)
        problems += [
            f"{issue.symbol} {issue.check}: {issue.detail}"
            for issue in check_bars(symbol, desk.bars.read(desk.provider.name, symbol), previous)
            if issue.severity is Severity.ERROR or issue.check == "stale"
        ]
    return problems


def close_ledger(desk: Desk, through: date, *, save: bool) -> tuple[LedgerDay, list[str]]:
    day = desk.state.latest_day()
    lines: list[str] = []
    if day is None:
        if save:
            raise ValueError("paper trading is not initialized; run `quant paper init` first")
        day = opening_day(through, desk.broker.equity())
        lines.append("not initialized yet: planning from a flat account")
    terms = desk.trading.terms
    closing = sessions(day.date + timedelta(days=1), through)
    benchmark = rates_on(closing, desk.rates.read())
    for session, rate in zip(closing, benchmark, strict=True):
        fills = session_fills(refresh_orders(desk, session), terms)
        bars = session_bars(desk, session, set(day.positions) | {fill.symbol for fill in fills})
        day = close_day(day, session, bars, fills, carry_for(terms, float(rate), day.equity))
        if save:
            desk.state.save_day(day)
        lines += [describe_fill(session, fill, bars[fill.symbol]) for fill in fills]
    lines.append(
        f"ledger {day.date}: equity {day.equity:,.2f}, drawdown {day.drawdown:.1%}, "
        f"{len(day.positions)} positions"
    )
    return day, lines


def refresh_orders(desk: Desk, session: date) -> list[OrderRecord]:
    refreshed: list[OrderRecord] = []
    for saved in desk.state.orders(session):
        if saved.status in TERMINAL_STATUSES:
            refreshed.append(saved)
            continue
        order = desk.broker.order(saved.client_order_id)
        if order is None:
            raise RuntimeError(f"order {saved.client_order_id} is missing at the broker")
        updated = replace(
            saved,
            status=order.status,
            filled_quantity=order.filled_quantity,
            filled_price=order.filled_price,
            commission=order.commission,
        )
        desk.state.save_order(updated)
        refreshed.append(updated)
    return refreshed


def session_targets(desk: Desk, day: date) -> dict[str, Target]:
    rows = desk.strategy.targets(market_prices(desk)).filter(pl.col("date") == day)
    if missing := sorted(set(desk.strategy.symbols) - set(rows["symbol"])):
        raise ValueError(f"no targets for {day} for {', '.join(missing)}")
    return {row["symbol"]: Target(row["weight"], row["band"]) for row in rows.iter_rows(named=True)}


def session_bars(desk: Desk, session: date, symbols: Iterable[str]) -> dict[str, Bar]:
    bars: dict[str, Bar] = {}
    for symbol in sorted(symbols):
        history = desk.bars.read(desk.provider.name, symbol)
        rows = pl.DataFrame() if history is None else history.filter(pl.col("date") == session)
        if rows.is_empty():
            raise ValueError(f"no {symbol} bar for {session}")
        row = rows.row(0, named=True)
        bars[symbol] = Bar(row["open"], row["close"], row["dividend"], row["split_factor"])
    return bars


def reconcile(expected: Mapping[str, float], actual: Mapping[str, float]) -> list[str]:
    return [
        f"{symbol}: expected {expected.get(symbol, 0.0):g}, broker {actual.get(symbol, 0.0):g}"
        for symbol in sorted(set(expected) | set(actual))
        if abs(expected.get(symbol, 0.0) - actual.get(symbol, 0.0)) >= POSITION_TOLERANCE
    ]


def submit_orders(desk: Desk, session: date, intents: Iterable[OrderIntent]) -> list[Rejection]:
    refused: list[Rejection] = []
    for intent in intents:
        try:
            desk.state.save_order(submit_intent(desk.broker, session, intent))
        except OrderRejectedError as error:
            refused.append(Rejection(intent, f"broker rejected the order: {error}"))
    return refused


def submit_intent(broker: Broker, session: date, intent: OrderIntent) -> OrderRecord:
    client_order_id = f"quant-{session:%Y%m%d}-{intent.symbol}"
    order = broker.order(client_order_id) or broker.submit_opening_order(
        client_order_id, intent.symbol, intent.quantity
    )
    return order_record(session, intent, order)


def order_record(session: date, intent: OrderIntent, order: BrokerOrder) -> OrderRecord:
    return OrderRecord(
        client_order_id=order.client_order_id,
        session=session,
        symbol=intent.symbol,
        quantity=intent.quantity,
        reference_price=intent.price,
        status=order.status,
        filled_quantity=order.filled_quantity,
        filled_price=order.filled_price,
        commission=order.commission,
    )


def describe_intent(intent: OrderIntent, submitted: bool = False) -> str:
    side = "buy" if intent.quantity > 0 else "sell"
    value = abs(intent.quantity) * intent.price
    suffix = " (already submitted)" if submitted else ""
    holding = f"holding {intent.current:g}, ~{value:,.0f}"
    return f"{intent.symbol} {side} {abs(intent.quantity)} ({holding}){suffix}"


def describe_order(order: OrderRecord) -> str:
    price = "" if order.filled_price is None else f" @ {order.filled_price:,.2f}"
    return f"{order.symbol} {order.status}: {order.filled_quantity:g} of {order.quantity:g}{price}"


def carry_for(terms: BrokerTerms, benchmark: float, equity: float) -> Carry:
    rate = np.array([benchmark])
    return Carry(
        borrow_fee=terms.borrow_fee,
        debit_rate=float(terms.debit_rate(rate)[0]),
        credit_rate=float(terms.credit_rate(rate, equity)[0]),
        credit_threshold=terms.cash_threshold,
    )


def session_fills(orders: Iterable[OrderRecord], terms: BrokerTerms) -> list[Fill]:
    return [
        Fill(
            order.symbol,
            order.filled_quantity,
            order.filled_price,
            fill_commission(order.filled_quantity, order.filled_price, order.commission, terms),
        )
        for order in orders
        if order.filled_quantity and order.filled_price is not None
    ]


def fill_commission(
    quantity: float, price: float, reported: float | None, terms: BrokerTerms
) -> float:
    if reported is not None:
        return reported
    return float(terms.commissions(np.array([quantity]), np.array([price]))[0])


def slippage_bps(fill: Fill, reference: float) -> float:
    return math.copysign(1.0, fill.quantity) * (fill.price / reference - 1.0) * BASIS_POINTS


def describe_fill(session: date, fill: Fill, bar: Bar) -> str:
    return (
        f"{session} {fill.symbol} {fill.quantity:+g} @ {fill.price:,.2f}, "
        f"open {bar.open:,.2f}, slippage {slippage_bps(fill, bar.open):+.1f}bp, "
        f"commission {fill.commission:,.2f}"
    )
