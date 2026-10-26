from dataclasses import replace
from datetime import UTC, date, datetime, time
from pathlib import Path

import polars as pl
import pytest
from factories import flat_rates, make_bars, random_walk, trading_with
from fakes import FakeBroker, FakeMarket, FakeProvider

from quant.core.calendar import at_new_york_time, sessions
from quant.core.trading import BrokerTerms
from quant.data.rates import RateStore
from quant.data.store import BarStore
from quant.live.jobs import (
    Desk,
    JobReport,
    broker_check,
    initialize,
    post_open,
    pre_open,
    replay_returns,
    tracking_report,
    upcoming_session,
)
from quant.live.state import OrderRecord, StateStore
from quant.strategies.fixed import FixedWeights

STRATEGY = FixedWeights(weights={"EQ1": 0.5, "EQ2": 0.4, "BND": -0.3, "GLD": 0.2})
TRADING = trading_with(
    BrokerTerms(borrow_fee=0.005, margin_spread=0.015), slippage_bps=3.0, max_order_fraction=1.0
)
DATES = sessions(date(2024, 1, 2), date(2024, 12, 31))[:60]
LAST, FIRST, SECOND = DATES[-3], DATES[-2], DATES[-1]


class FlakyState(StateStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.failures = 0

    def save_order(self, order: OrderRecord) -> None:
        if self.failures:
            self.failures -= 1
            raise OSError("disk full")
        super().save_order(order)


def gapless(bars: pl.DataFrame) -> pl.DataFrame:
    return bars.with_columns(
        pl.col("close").shift(1).fill_null(pl.col("close")).alias("open")
    ).with_columns(
        (pl.max_horizontal("open", "close") * 1.01).alias("high"),
        (pl.min_horizontal("open", "close") * 0.99).alias("low"),
    )


def build_desk(tmp_path: Path) -> Desk:
    broker = FakeBroker()
    broker.borrowable = set(STRATEGY.symbols)
    bars = {
        symbol: gapless(make_bars(random_walk(seed, len(DATES))))
        for seed, symbol in enumerate(STRATEGY.symbols)
    }
    rates = RateStore(tmp_path / "data")
    rates.write(flat_rates(0.04))
    return Desk(
        bars=BarStore(tmp_path / "data"),
        rates=rates,
        provider=FakeProvider(bars, through=LAST),
        broker=broker,
        market=FakeMarket(),
        state=FlakyState(tmp_path / "state.sqlite3"),
        strategy=STRATEGY,
        trading=TRADING,
    )


@pytest.fixture
def new_desk(tmp_path: Path) -> Desk:
    return build_desk(tmp_path)


@pytest.fixture
def desk(new_desk: Desk) -> Desk:
    assert initialize(new_desk, at_new_york_time(LAST, time(20, 0)), None).ok
    return new_desk


def broker(desk: Desk) -> FakeBroker:
    assert isinstance(desk.broker, FakeBroker)
    return desk.broker


def market(desk: Desk) -> FakeMarket:
    assert isinstance(desk.market, FakeMarket)
    return desk.market


def provider(desk: Desk) -> FakeProvider:
    assert isinstance(desk.provider, FakeProvider)
    return desk.provider


def morning(day: date) -> datetime:
    return at_new_york_time(day, time(8, 45))


def prices_on(desk: Desk, day: date, column: str) -> dict[str, float]:
    return {
        symbol: bars.filter(pl.col("date") == day)[column][0]
        for symbol, bars in provider(desk).bars.items()
    }


def trade_first_session(desk: Desk, markup: float = 0.0) -> JobReport:
    report = pre_open(desk, FIRST, morning(FIRST), submit=True)
    assert report.ok, report.text
    opens = prices_on(desk, FIRST, "open")
    broker(desk).open_auction({symbol: price * (1.0 + markup) for symbol, price in opens.items()})
    provider(desk).through = FIRST
    return report


def test_first_session_submits_whole_share_opening_orders(desk: Desk) -> None:
    report = pre_open(desk, FIRST, morning(FIRST), submit=True)

    assert report.ok, report.text
    orders = desk.state.orders(FIRST)
    assert orders
    assert {order.client_order_id for order in orders} == set(broker(desk).orders)
    for order in orders:
        assert order.client_order_id == f"quant-{FIRST:%Y%m%d}-{order.symbol}"
        assert order.quantity == int(order.quantity) != 0


def test_rerunning_pre_open_does_not_duplicate_orders(desk: Desk) -> None:
    pre_open(desk, FIRST, morning(FIRST), submit=True)
    submitted = len(broker(desk).orders)

    report = pre_open(desk, FIRST, morning(FIRST), submit=True)

    assert len(broker(desk).orders) == submitted
    assert "(already submitted)" in report.text


def test_orders_already_at_the_broker_are_recorded_after_a_crash(desk: Desk) -> None:
    assert isinstance(desk.state, FlakyState)
    desk.state.failures = 1
    with pytest.raises(OSError, match="disk full"):
        pre_open(desk, FIRST, morning(FIRST), submit=True)
    assert len(broker(desk).orders) == 1

    report = pre_open(desk, FIRST, morning(FIRST), submit=True)

    assert report.ok, report.text
    assert {order.client_order_id for order in desk.state.orders(FIRST)} == set(broker(desk).orders)


def test_post_open_records_fills_and_matches_the_broker(desk: Desk) -> None:
    trade_first_session(desk)

    report = post_open(desk, FIRST)

    assert report.ok, report.text
    assert {order.status for order in desk.state.orders(FIRST)} == {"filled"}


def test_the_ledger_books_commissions_reported_by_the_broker(desk: Desk) -> None:
    assert pre_open(desk, FIRST, morning(FIRST), submit=True).ok
    broker(desk).open_auction(prices_on(desk, FIRST, "open"), commission=1.25)
    provider(desk).through = FIRST

    report = pre_open(desk, SECOND, morning(SECOND), submit=True)

    fills = [line for line in report.lines if line.startswith(str(FIRST))]
    assert fills
    assert all(line.endswith("commission 1.25") for line in fills)


def test_the_ledger_models_commissions_the_broker_does_not_report(desk: Desk) -> None:
    terms = BrokerTerms(commission_per_share=0.0035, commission_minimum=0.35, fees_per_share=0.0017)
    tiered = replace(desk, trading=trading_with(terms, max_order_fraction=1.0))
    trade_first_session(tiered)

    report = pre_open(tiered, SECOND, morning(SECOND), submit=True)

    (order,) = [order for order in tiered.state.orders(FIRST) if order.symbol == "EQ1"]
    shares = abs(order.filled_quantity)
    expected = max(0.35, 0.0035 * shares) + 0.0017 * shares
    assert f"commission {expected:,.2f}" in report.text


def test_post_open_needs_the_ledger_to_end_on_the_previous_session(desk: Desk) -> None:
    trade_first_session(desk)
    pre_open(desk, SECOND, morning(SECOND), submit=True)

    report = post_open(desk, FIRST)

    assert not report.ok
    assert f"the ledger ends on {FIRST}, expected {LAST}" in report.text


def test_next_session_closes_the_ledger_from_the_fills(desk: Desk) -> None:
    trade_first_session(desk)

    report = pre_open(desk, SECOND, morning(SECOND), submit=True)

    assert report.ok, report.text
    day = desk.state.latest_day()
    assert day is not None
    assert day.date == FIRST
    assert day.positions == broker(desk).positions()
    closes = prices_on(desk, FIRST, "close")
    marked = sum(shares * closes[symbol] for symbol, shares in day.positions.items())
    assert day.equity == pytest.approx(day.cash + marked)
    assert "slippage +0.0bp" in report.text


def test_tracking_compares_the_ledger_with_a_backtest_replay(desk: Desk) -> None:
    trade_first_session(desk)
    pre_open(desk, SECOND, morning(SECOND), submit=True)

    report = tracking_report(desk)

    assert report.ok
    assert f"1 trading days since {LAST}" in report.text
    assert "backtest replay" in report.text


def test_the_replay_matches_the_paper_ledger(desk: Desk) -> None:
    free = replace(desk, trading=trading_with(max_order_fraction=1.0))
    trade_first_session(free)
    pre_open(free, SECOND, morning(SECOND), submit=True)

    start, first = free.state.days()
    replay = replay_returns(free, start.date).filter(pl.col("date") == FIRST)
    paper = first.equity / start.equity - 1.0
    assert abs(paper) > 1e-3
    assert paper == pytest.approx(replay["replay"][0], abs=1e-4)


def test_tracking_summarises_slippage_against_the_open(desk: Desk) -> None:
    trade_first_session(desk, markup=0.0005)
    pre_open(desk, SECOND, morning(SECOND), submit=True)

    report = tracking_report(desk)

    assert "slippage against the open in bps, positive is worse:" in report.lines
    rows = {
        line.split()[0]: line.split()[1:]
        for line in report.lines
        if line.startswith(("EQ1", "BND", "all"))
    }
    assert rows["EQ1"] == ["1", "+5.0", "+5.0", "3.0"]
    assert rows["BND"] == ["1", "-5.0", "-5.0", "3.0"]
    assert rows["all"][2:] == ["+5.0", "3.0"]


def test_a_broker_rejection_does_not_stop_the_other_orders(desk: Desk) -> None:
    broker(desk).refusals = {"EQ1": "insufficient buying power"}

    report = pre_open(desk, FIRST, morning(FIRST), submit=True)

    assert not report.ok
    assert "broker rejected the order: insufficient buying power" in report.text
    sent = {order.symbol for order in desk.state.orders(FIRST)}
    assert sent
    assert "EQ1" not in sent
    broker(desk).refusals = {}
    retry = pre_open(desk, FIRST, morning(FIRST), submit=True)
    assert retry.ok, retry.text
    assert {order.symbol for order in desk.state.orders(FIRST)} == sent | {"EQ1"}


def test_non_sessions_are_skipped(desk: Desk) -> None:
    saturday = date(2024, 7, 6)

    report = pre_open(desk, saturday, morning(saturday), submit=True)

    assert report.ok
    assert report.lines == ["not a trading session"]


@pytest.mark.parametrize(
    "now", [at_new_york_time(LAST, time(18, 59)), at_new_york_time(FIRST, time(9, 28))]
)
def test_orders_are_only_sent_inside_the_submission_window(desk: Desk, now: datetime) -> None:
    report = pre_open(desk, FIRST, now, submit=True)

    assert not report.ok
    assert "outside the submission window" in report.text
    assert not broker(desk).orders


def test_orders_can_be_queued_the_evening_before(desk: Desk) -> None:
    report = pre_open(desk, FIRST, at_new_york_time(LAST, time(19, 0)), submit=True)

    assert report.ok, report.text
    assert desk.state.orders(FIRST)


@pytest.mark.parametrize(
    ("now", "session"),
    [
        (at_new_york_time(date(2026, 10, 26), time(8, 45)), date(2026, 10, 26)),
        (at_new_york_time(date(2026, 10, 26), time(9, 28)), date(2026, 10, 27)),
        (datetime(2026, 10, 27, 1, 0, tzinfo=UTC), date(2026, 10, 27)),
        (at_new_york_time(date(2026, 10, 25), time(7, 0)), date(2026, 10, 26)),
        (at_new_york_time(date(2026, 11, 25), time(20, 0)), date(2026, 11, 27)),
    ],
)
def test_the_upcoming_session_is_the_next_opening_auction(now: datetime, session: date) -> None:
    assert upcoming_session(now) == session


def test_stale_data_stops_trading(desk: Desk) -> None:
    provider(desk).through = DATES[-4]

    report = pre_open(desk, FIRST, morning(FIRST), submit=True)

    assert not report.ok
    assert "market data is not ready" in report.text
    assert not broker(desk).orders


def test_a_position_mismatch_stops_trading(desk: Desk) -> None:
    broker(desk).held["EQ1"] = 5.0

    report = pre_open(desk, FIRST, morning(FIRST), submit=True)

    assert not report.ok
    assert "positions differ from the broker" in report.text
    assert not broker(desk).orders


def test_rejected_orders_are_reported_and_not_sent(desk: Desk) -> None:
    strict = replace(desk, trading=trading_with(max_order_fraction=0.001))

    report = pre_open(strict, FIRST, morning(FIRST), submit=True)

    assert not report.ok
    assert "rejected" in report.text
    assert not broker(desk).orders


def test_symbols_splitting_at_the_open_are_held_back(desk: Desk) -> None:
    market(desk).announced_splits = {
        ("EQ1", FIRST): "2-for-1 split",
        ("EQ2", SECOND): "3-for-1 split",
    }

    report = pre_open(desk, FIRST, morning(FIRST), submit=True)

    assert not report.ok
    assert "2-for-1 split takes effect at this open" in report.text
    traded = {order.symbol for order in desk.state.orders(FIRST)}
    assert "EQ1" not in traded
    assert "EQ2" in traded


def test_dry_runs_submit_nothing_and_leave_the_ledger_alone(desk: Desk) -> None:
    trade_first_session(desk)
    submitted = len(broker(desk).orders)

    report = pre_open(desk, SECOND, morning(SECOND), submit=False)

    assert report.ok, report.text
    assert report.title.endswith("(dry run)")
    assert f"ledger {FIRST}" in report.text
    assert len(broker(desk).orders) == submitted
    assert desk.state.orders(SECOND) == []
    assert [day.date for day in desk.state.days()] == [LAST]


def test_dry_runs_work_before_initialization(new_desk: Desk) -> None:
    report = pre_open(new_desk, FIRST, morning(FIRST), submit=False)

    assert report.ok, report.text
    assert "not initialized yet" in report.text
    assert f"ledger {LAST}: equity 100,000.00" in report.text
    assert new_desk.state.latest_day() is None


def test_submitting_needs_initialization(new_desk: Desk) -> None:
    with pytest.raises(ValueError, match="not initialized"):
        pre_open(new_desk, FIRST, morning(FIRST), submit=True)

    assert not broker(new_desk).orders


def test_initialization_happens_once(desk: Desk) -> None:
    report = initialize(desk, morning(FIRST), None)

    assert not report.ok
    assert "already initialized" in report.text


def test_initialization_needs_a_flat_account(new_desk: Desk) -> None:
    broker(new_desk).held["EQ1"] = 1.0

    report = initialize(new_desk, morning(FIRST), None)

    assert not report.ok
    assert "must be flat" in report.text


def test_the_broker_check_lists_blocked_shorts_and_missing_prices(new_desk: Desk) -> None:
    broker(new_desk).borrowable = {"EQ1", "EQ2"}
    market(new_desk).prices = {"EQ1": 1.0, "EQ2": 1.0, "BND": 1.0}

    report = broker_check(new_desk)

    assert not report.ok
    assert "cannot short: BND, GLD" in report.lines
    assert "no latest price for GLD" in report.lines


def test_orders_below_the_minimum_are_listed_but_not_sent(desk: Desk) -> None:
    minimum = replace(desk, trading=TRADING.model_copy(update={"min_order_value": 1e9}))

    report = pre_open(minimum, FIRST, morning(FIRST), submit=True)

    assert report.ok, report.text
    assert "below the $1,000,000,000 minimum order, not traded: BND, EQ1, EQ2, GLD" in report.lines
    assert not broker(desk).orders
