from collections.abc import Mapping, Sequence
from datetime import date

import pytest

from quant.live.ledger import Bar, Carry, Fill, LedgerDay, close_day, opening_day

DAY = date(2026, 10, 26)
NO_CARRY = Carry()


def bar(close: float, *, dividend: float = 0.0, split: float = 1.0) -> Bar:
    return Bar(open=close, close=close, dividend=dividend, split_factor=split)


def close(
    previous: LedgerDay,
    bars: Mapping[str, Bar],
    fills: Sequence[Fill] = (),
    carry: Carry = NO_CARRY,
) -> LedgerDay:
    return close_day(previous, DAY, bars, fills, carry)


def test_fills_move_cash_into_positions_marked_at_the_close() -> None:
    day = close(
        opening_day(date(2026, 10, 23), 10_000.0),
        {"SPY": bar(101.0)},
        [Fill("SPY", 10.0, 100.0)],
    )

    assert day.cash == pytest.approx(9_000.0)
    assert day.positions == {"SPY": 10.0}
    assert day.equity == pytest.approx(10_010.0)


def test_commissions_come_out_of_cash() -> None:
    day = close(
        opening_day(date(2026, 10, 23), 10_000.0),
        {"SPY": bar(100.0)},
        [Fill("SPY", 10.0, 100.0, commission=0.35)],
    )

    assert day.cash == pytest.approx(9_000.0 - 0.35)


def test_dividends_are_paid_on_shares_held_before_the_ex_date() -> None:
    previous = LedgerDay(date(2026, 10, 23), 0.0, {"SPY": 10.0, "TLT": -5.0}, 1_000.0, 1_000.0)

    day = close(previous, {"SPY": bar(100.0, dividend=2.0), "TLT": bar(100.0, dividend=1.0)})

    assert day.cash == pytest.approx(20.0 - 5.0)


def test_splits_change_the_share_count() -> None:
    previous = LedgerDay(date(2026, 10, 23), 0.0, {"USO": 80.0}, 170.4, 170.4)

    day = close(previous, {"USO": bar(18.0, split=0.125)})

    assert day.positions == {"USO": 10.0}
    assert day.equity == pytest.approx(180.0)


def test_closed_positions_are_dropped() -> None:
    previous = LedgerDay(date(2026, 10, 23), 0.0, {"SPY": 10.0}, 1_000.0, 1_000.0)

    day = close(previous, {"SPY": bar(100.0)}, [Fill("SPY", -10.0, 100.0)])

    assert day.positions == {}
    assert day.cash == pytest.approx(1_000.0)


def test_short_positions_and_borrowed_cash_cost_carry() -> None:
    previous = LedgerDay(
        date(2026, 10, 23), -1_000.0, {"SPY": 30.0, "TLT": -10.0}, 1_000.0, 1_000.0
    )

    day = close(
        previous,
        {"SPY": bar(100.0), "TLT": bar(100.0)},
        carry=Carry(borrow_fee=0.252, debit_rate=0.252),
    )

    assert day.cash == pytest.approx(-1_000.0 - 1.0 - 1.0)


def test_idle_cash_above_the_threshold_earns_interest_but_short_proceeds_do_not() -> None:
    previous = LedgerDay(date(2026, 10, 23), 30_000.0, {"TLT": -100.0}, 20_000.0, 20_000.0)

    day = close(
        previous,
        {"TLT": bar(100.0)},
        carry=Carry(credit_rate=0.252, credit_threshold=10_000.0),
    )

    assert day.cash == pytest.approx(30_000.0 + 0.252 * (30_000 - 10_000 - 10_000) / 252)


def test_drawdown_is_measured_from_the_peak() -> None:
    previous = LedgerDay(date(2026, 10, 23), 0.0, {"SPY": 10.0}, 1_000.0, 1_200.0)

    day = close(previous, {"SPY": bar(90.0)})

    assert day.peak_equity == 1_200.0
    assert day.drawdown == pytest.approx(900.0 / 1_200.0 - 1.0)
