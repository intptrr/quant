from dataclasses import replace
from datetime import date
from pathlib import Path

from quant.live.ledger import LedgerDay
from quant.live.state import OrderRecord, StateStore

ORDER = OrderRecord(
    client_order_id="quant-20261026-SPY",
    session=date(2026, 10, 26),
    symbol="SPY",
    quantity=10.0,
    reference_price=500.0,
    status="accepted",
    filled_quantity=0.0,
    filled_price=None,
)


def test_ledger_days_round_trip_in_date_order(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.sqlite3")
    later = LedgerDay(date(2026, 10, 26), 5_000.0, {"SPY": 10.0}, 10_010.0, 10_010.0)
    earlier = LedgerDay(date(2026, 10, 23), 10_000.0, {}, 10_000.0, 10_000.0)

    state.save_day(later)
    state.save_day(earlier)

    assert state.days() == [earlier, later]
    assert state.latest_day() == later


def test_an_empty_store_has_no_latest_day(tmp_path: Path) -> None:
    assert StateStore(tmp_path / "state.sqlite3").latest_day() is None


def test_orders_are_updated_in_place(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.sqlite3")
    state.save_order(ORDER)

    filled = replace(ORDER, status="filled", filled_quantity=10.0, filled_price=501.0)
    state.save_order(filled)

    assert state.orders(date(2026, 10, 26)) == [filled]
    assert state.orders(date(2026, 10, 27)) == []
