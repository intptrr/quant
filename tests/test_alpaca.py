from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from alpaca.common.exceptions import APIError
from alpaca.data.models.corporate_actions import CorporateActionsSet
from alpaca.trading.models import Order, Position

from quant.execution.alpaca import (
    AlpacaBroker,
    describe_splits,
    expect,
    signed_quantity,
    to_broker_order,
)
from quant.execution.broker import OrderRejectedError

ACTIONS = CorporateActionsSet(
    {
        "forward_splits": [
            {
                "id": "0b6f4cf8-9c0a-4a51-9b1c-2f1f5b7c1a01",
                "symbol": "QUAL",
                "cusip": "46432F339",
                "new_rate": 4.0,
                "old_rate": 1.0,
                "process_date": "2026-11-05",
                "ex_date": "2026-11-05",
                "payable_date": "2026-11-04",
            }
        ],
        "reverse_splits": [
            {
                "id": "0b6f4cf8-9c0a-4a51-9b1c-2f1f5b7c1a02",
                "symbol": "USO",
                "old_cusip": "91232N108",
                "new_cusip": "91232N207",
                "new_rate": 1.0,
                "old_rate": 8.0,
                "process_date": "2020-04-29",
                "ex_date": "2020-04-29",
            }
        ],
        "stock_dividends": [
            {
                "id": "0b6f4cf8-9c0a-4a51-9b1c-2f1f5b7c1a03",
                "symbol": "XYZ",
                "cusip": "000000000",
                "rate": 0.05,
                "process_date": "2026-11-05",
                "ex_date": "2026-11-05",
            }
        ],
    }
)

ORDER: dict[str, Any] = {
    "id": "61e69015-8549-4bfd-b9c3-01e75843f47d",
    "client_order_id": "quant-20261026-SPY",
    "created_at": "2026-10-26T12:45:00Z",
    "updated_at": "2026-10-26T13:30:01Z",
    "submitted_at": "2026-10-26T12:45:00Z",
    "asset_id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
    "symbol": "SPY",
    "asset_class": "us_equity",
    "qty": "10",
    "filled_qty": "4",
    "filled_avg_price": "500.25",
    "order_class": "",
    "order_type": "market",
    "type": "market",
    "side": "sell",
    "time_in_force": "opg",
    "status": "partially_filled",
    "extended_hours": False,
}
POSITION: dict[str, Any] = {
    "asset_id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
    "symbol": "TLT",
    "exchange": "NASDAQ",
    "asset_class": "us_equity",
    "avg_entry_price": "90.0",
    "cost_basis": "1080",
}


def test_sell_orders_have_negative_quantities() -> None:
    order = to_broker_order(Order.model_validate(ORDER))

    assert (order.quantity, order.filled_quantity, order.filled_price) == (-10.0, -4.0, 500.25)
    assert (order.status, order.done) == ("partially_filled", False)


def test_unfilled_orders_have_no_fill_price() -> None:
    order = to_broker_order(
        Order.model_validate(
            {**ORDER, "side": "buy", "filled_qty": "0", "filled_avg_price": None, "status": "new"}
        )
    )

    assert (order.quantity, order.filled_quantity, order.filled_price) == (10.0, 0.0, None)


@pytest.mark.parametrize(
    ("quantity", "side", "signed"),
    [("12", "long", 12.0), ("-12", "short", -12.0), ("12", "short", -12.0)],
)
def test_short_positions_are_negative(quantity: str, side: str, signed: float) -> None:
    position = Position.model_validate({**POSITION, "qty": quantity, "side": side})

    assert signed_quantity(position) == signed


def test_unexpected_response_types_are_rejected() -> None:
    with pytest.raises(TypeError, match="expected Order from Alpaca, got dict"):
        expect({"raw": True}, Order)


def test_splits_are_described_when_their_ex_date_is_in_the_window() -> None:
    assert describe_splits(ACTIONS, date(2026, 11, 5), date(2026, 11, 5)) == {
        "QUAL": "4-for-1 split",
        "XYZ": "stock dividend of 0.05 shares per share",
    }
    assert describe_splits(ACTIONS, date(2020, 4, 28), date(2020, 4, 29)) == {
        "USO": "1-for-8 split"
    }
    assert describe_splits(ACTIONS, date(2026, 11, 6), date(2026, 11, 9)) == {}


class FakeHTTPError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.response = SimpleNamespace(status_code=status)


def failing_broker(monkeypatch: pytest.MonkeyPatch, status: int) -> AlpacaBroker:
    broker = AlpacaBroker("key", "secret")
    error = APIError(
        '{"code": 40310000, "message": "insufficient buying power"}', FakeHTTPError(status)
    )

    def submit_order(order_data: object) -> object:
        raise error

    monkeypatch.setattr(broker._trading, "submit_order", submit_order)
    return broker


def test_refused_orders_raise_order_rejected_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = failing_broker(monkeypatch, 403)

    with pytest.raises(OrderRejectedError, match="insufficient buying power"):
        broker.submit_opening_order("quant-20261026-SPY", "SPY", 10)


def test_other_api_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = failing_broker(monkeypatch, 500)

    with pytest.raises(APIError):
        broker.submit_opening_order("quant-20261026-SPY", "SPY", 10)
