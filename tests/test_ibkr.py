import math
from datetime import UTC, datetime
from typing import cast

import pytest
from ib_async import (
    IB,
    AccountValue,
    CommissionReport,
    Contract,
    Execution,
    ExecutionFilter,
    Fill,
    Order,
    OrderStatus,
    Position,
    Stock,
    Ticker,
    Trade,
    TradeLogEntry,
)

from quant.execution.broker import OrderRejectedError
from quant.execution.ibkr import IbkrBroker

ACCOUNT = "DU1234567"
REF = "quant-20261026-SPY"


def trade(
    status: str, *, action: str = "SELL", quantity: float = 10, filled: float | None = None
) -> Trade:
    order = Order(action=action, totalQuantity=quantity, orderRef=REF, account=ACCOUNT)
    if filled is not None:
        order.filledQuantity = filled
    return Trade(Stock("SPY", "SMART", "USD"), order, OrderStatus(status=status))


def fill(exec_id: str, shares: float, price: float, *, commission: bool = True) -> Fill:
    report = CommissionReport(execId=exec_id, commission=0.6) if commission else CommissionReport()
    execution = Execution(execId=exec_id, shares=shares, price=price, side="SLD", orderRef=REF)
    return Fill(Stock("SPY", "SMART", "USD"), execution, report, datetime.now(UTC))


class FakeIB:
    def __init__(self) -> None:
        self.open: list[Trade] = []
        self.completed: list[Trade] = []
        self.executions: list[Fill] = []
        self.placed: list[tuple[Contract, Order]] = []
        self.placed_status = "PreSubmitted"
        self.placed_log: list[TradeLogEntry] = []
        self.shortable: dict[str, float] = {}
        self.cancelled: list[Contract] = []
        self.market_data_type = 1
        self.disconnected = False

    def isConnected(self) -> bool:
        return True

    def managedAccounts(self) -> list[str]:
        return [ACCOUNT]

    def accountSummary(self, account: str = "") -> list[AccountValue]:
        return [
            AccountValue(account, "NetLiquidation", "99000.5", "USD", ""),
            AccountValue(account, "NetLiquidation", "1", "BASE", ""),
        ]

    def positions(self, account: str = "") -> list[Position]:
        return [
            Position(account, Stock("SPY", "SMART", "USD"), 10.0, 500.0),
            Position(account, Stock("TLT", "SMART", "USD"), -20.0, 90.0),
            Position(account, Contract(secType="CASH", symbol="EUR"), 5.0, 1.1),
        ]

    def reqAllOpenOrders(self) -> list[Trade]:
        return self.open

    def reqCompletedOrders(self, apiOnly: bool) -> list[Trade]:
        return self.completed

    def reqExecutions(self, execFilter: ExecutionFilter | None = None) -> list[Fill]:
        return self.executions

    def qualifyContracts(self, *contracts: Contract) -> list[Contract]:
        return [] if contracts[0].symbol == "NOPE" else list(contracts)

    def placeOrder(self, contract: Contract, order: Order) -> Trade:
        self.placed.append((contract, order))
        return Trade(contract, order, OrderStatus(status=self.placed_status), log=self.placed_log)

    def sleep(self, secs: float = 0.02) -> bool:
        return True

    def reqMktData(self, contract: Contract, genericTickList: str = "") -> Ticker:
        ticker = Ticker(contract=contract)
        ticker.shortable = self.shortable.get(contract.symbol, math.nan)
        return ticker

    def cancelMktData(self, contract: Contract) -> bool:
        self.cancelled.append(contract)
        return True

    def reqMarketDataType(self, marketDataType: int) -> None:
        self.market_data_type = marketDataType

    def disconnect(self) -> None:
        self.disconnected = True


@pytest.fixture
def ib() -> FakeIB:
    return FakeIB()


def broker(ib: FakeIB) -> IbkrBroker:
    return IbkrBroker("127.0.0.1", 4002, 17, ACCOUNT, ib=cast(IB, ib))


def test_equity_is_the_usd_net_liquidation_value(ib: FakeIB) -> None:
    assert broker(ib).equity() == 99000.5


def test_positions_are_signed_stock_quantities(ib: FakeIB) -> None:
    assert broker(ib).positions() == {"SPY": 10.0, "TLT": -20.0}


def test_opening_orders_are_market_on_open_orders_tagged_with_our_id(ib: FakeIB) -> None:
    order = broker(ib).submit_opening_order(REF, "SPY", -10)

    ((contract, placed),) = ib.placed
    assert (contract.symbol, contract.exchange) == ("SPY", "SMART")
    assert (placed.action, placed.totalQuantity, placed.orderType, placed.tif) == (
        "SELL",
        10,
        "MKT",
        "OPG",
    )
    assert (placed.orderRef, placed.account) == (REF, ACCOUNT)
    assert (order.quantity, order.status, order.filled_price) == (-10.0, "accepted", None)


def test_refused_orders_raise_with_ibkr_messages(ib: FakeIB) -> None:
    ib.placed_status = "Inactive"
    ib.placed_log = [
        TradeLogEntry(datetime.now(UTC), "Inactive", "Order rejected - reason: no shares", 201)
    ]

    with pytest.raises(OrderRejectedError, match="no shares"):
        broker(ib).submit_opening_order(REF, "SPY", -10)


def test_unknown_symbols_are_an_error(ib: FakeIB) -> None:
    with pytest.raises(ValueError, match="does not know the US stock NOPE"):
        broker(ib).submit_opening_order(REF, "NOPE", 1)


def test_orders_are_found_by_our_id(ib: FakeIB) -> None:
    ib.open = [trade("PreSubmitted")]

    order = broker(ib).order(REF)

    assert order is not None
    assert (order.status, order.filled_quantity, order.commission) == ("accepted", 0.0, None)
    assert broker(ib).order("quant-20261026-QQQ") is None


def test_filled_orders_use_executions_for_price_and_commission(ib: FakeIB) -> None:
    ib.completed = [trade("Filled", filled=10)]
    ib.executions = [fill("e1", 4, 500.0), fill("e2", 6, 501.0)]

    order = broker(ib).order(REF)

    assert order is not None
    assert order.status == "filled"
    assert order.filled_quantity == -10.0
    assert order.filled_price == pytest.approx(500.6)
    assert order.commission == pytest.approx(1.2)


def test_partly_filled_auction_orders_end_cancelled_with_their_fills(ib: FakeIB) -> None:
    ib.completed = [trade("Cancelled", filled=4)]
    ib.executions = [fill("e1", 4, 500.0, commission=False)]

    order = broker(ib).order(REF)

    assert order is not None
    assert (order.status, order.filled_quantity, order.commission) == ("canceled", -4.0, None)


def test_fills_that_ibkr_no_longer_reports_are_an_error(ib: FakeIB) -> None:
    ib.completed = [trade("Filled", filled=10)]

    with pytest.raises(RuntimeError, match="only keeps the current day's executions"):
        broker(ib).order(REF)


def test_short_availability_comes_from_market_data(ib: FakeIB) -> None:
    ib.shortable = {"TLT": 3.0, "FXE": 2.0}

    assert broker(ib).shortable(["TLT", "FXE"]) == {"TLT": True, "FXE": False}
    assert [contract.symbol for contract in ib.cancelled] == ["TLT", "FXE"]


def test_delayed_market_data_is_used_without_a_subscription(ib: FakeIB) -> None:
    broker(ib)

    assert ib.market_data_type == 3


def test_missing_short_availability_is_an_error(ib: FakeIB) -> None:
    with pytest.raises(RuntimeError, match="no short availability for USO"):
        broker(ib).shortable(["USO"])


def test_the_account_must_be_on_the_login(ib: FakeIB) -> None:
    with pytest.raises(ValueError, match="DU0000000 is not on this login"):
        IbkrBroker("127.0.0.1", 4002, 17, "DU0000000", ib=cast(IB, ib))


def test_closing_disconnects(ib: FakeIB) -> None:
    broker(ib).close()

    assert ib.disconnected
