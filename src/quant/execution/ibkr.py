import math
from collections.abc import Iterable, Sequence

from ib_async import IB, Contract, ExecutionFilter, Fill, MarketOrder, Stock, Trade
from ib_async.util import UNSET_DOUBLE

from quant.execution.broker import BrokerOrder, OrderRejectedError

CONNECT_TIMEOUT = 10.0
SUBMIT_WAIT = 2.0
SHORTABLE_WAIT = 3.0
EASY_TO_BORROW = 2.5
DELAYED_IF_NOT_SUBSCRIBED = 3
STATUSES = {
    "Filled": "filled",
    "Cancelled": "canceled",
    "ApiCancelled": "canceled",
    "Inactive": "rejected",
    "ValidationError": "rejected",
}
REFUSED = frozenset({"Inactive", "Cancelled", "ApiCancelled", "ValidationError"})


class IbkrBroker:
    def __init__(
        self, host: str, port: int, client_id: int, account: str, *, ib: IB | None = None
    ) -> None:
        self._ib = ib or IB()
        self._account = account
        self._contracts: dict[str, Contract] = {}
        if not self._ib.isConnected():
            try:
                self._ib.connect(
                    host, port, clientId=client_id, timeout=CONNECT_TIMEOUT, account=account
                )
            except (ConnectionError, TimeoutError) as error:
                raise ConnectionError(
                    f"cannot reach IB Gateway at {host}:{port}; is it running and logged in?"
                ) from error
        accounts = self._ib.managedAccounts()
        if account not in accounts:
            raise ValueError(f"IBKR account {account} is not on this login ({', '.join(accounts)})")
        self._ib.reqMarketDataType(DELAYED_IF_NOT_SUBSCRIBED)

    def equity(self) -> float:
        values = [
            value
            for value in self._ib.accountSummary(self._account)
            if value.tag == "NetLiquidation" and value.currency == "USD"
        ]
        if not values:
            raise RuntimeError(f"IBKR reported no net liquidation value for {self._account}")
        return float(values[0].value)

    def positions(self) -> dict[str, float]:
        return {
            position.contract.symbol: float(position.position)
            for position in self._ib.positions(self._account)
            if position.contract.secType == "STK" and position.position
        }

    def open_orders(self) -> list[BrokerOrder]:
        return [
            to_broker_order(trade.order.orderRef, trade, [])
            for trade in self._ib.reqAllOpenOrders()
            if trade.order.account == self._account
        ]

    def order(self, client_order_id: str) -> BrokerOrder | None:
        trades = [
            trade
            for trade in (*self._ib.reqAllOpenOrders(), *self._ib.reqCompletedOrders(apiOnly=True))
            if trade.order.orderRef == client_order_id
        ]
        fills = [
            fill
            for fill in self._ib.reqExecutions(ExecutionFilter(acctCode=self._account))
            if fill.execution.orderRef == client_order_id
        ]
        if not trades and not fills:
            return None
        trade = trades[-1] if trades else None
        if trade is not None and reported_fill(trade) and not fills:
            raise RuntimeError(
                f"IBKR no longer reports the fills of {client_order_id}; it only keeps the "
                "current day's executions, so run `quant paper post-open` on the trading day"
            )
        return to_broker_order(client_order_id, trade, fills)

    def submit_opening_order(self, client_order_id: str, symbol: str, quantity: int) -> BrokerOrder:
        order = MarketOrder(
            "BUY" if quantity > 0 else "SELL",
            abs(quantity),
            tif="OPG",
            orderRef=client_order_id,
            account=self._account,
        )
        trade = self._ib.placeOrder(self._contract(symbol), order)
        self._ib.sleep(SUBMIT_WAIT)
        if trade.orderStatus.status in REFUSED:
            messages = [entry.message for entry in trade.log if entry.errorCode]
            raise OrderRejectedError("; ".join(messages) or trade.orderStatus.status)
        return to_broker_order(client_order_id, trade, [])

    def shortable(self, symbols: Sequence[str]) -> dict[str, bool]:
        contracts = {symbol: self._contract(symbol) for symbol in symbols}
        tickers = {
            symbol: self._ib.reqMktData(contract, genericTickList="236")
            for symbol, contract in contracts.items()
        }
        if tickers:
            self._ib.sleep(SHORTABLE_WAIT)
        result: dict[str, bool] = {}
        for symbol, ticker in tickers.items():
            self._ib.cancelMktData(contracts[symbol])
            if ticker.shortable is None or math.isnan(ticker.shortable):
                raise RuntimeError(
                    f"IBKR sent no short availability for {symbol}; "
                    "is market data shared with this account?"
                )
            result[symbol] = ticker.shortable > EASY_TO_BORROW
        return result

    def close(self) -> None:
        self._ib.disconnect()

    def _contract(self, symbol: str) -> Contract:
        if symbol not in self._contracts:
            qualified = self._ib.qualifyContracts(Stock(symbol, "SMART", "USD"))
            if not qualified:
                raise ValueError(f"IBKR does not know the US stock {symbol}")
            self._contracts[symbol] = qualified[0]
        return self._contracts[symbol]


def to_broker_order(
    client_order_id: str, trade: Trade | None, fills: Iterable[Fill]
) -> BrokerOrder:
    fills = list(fills)
    direction = order_direction(trade, fills)
    shares = sum(float(fill.execution.shares) for fill in fills)
    value = sum(float(fill.execution.shares) * float(fill.execution.price) for fill in fills)
    return BrokerOrder(
        client_order_id=client_order_id,
        symbol=trade.contract.symbol if trade is not None else fills[0].contract.symbol,
        quantity=direction * float(trade.order.totalQuantity if trade is not None else shares),
        filled_quantity=direction * shares,
        filled_price=value / shares if shares else None,
        status=order_status(trade, shares),
        commission=fill_commission(fills),
    )


def reported_fill(trade: Trade) -> float:
    quantity = float(trade.order.filledQuantity)
    return max(float(trade.orderStatus.filled), quantity if quantity < UNSET_DOUBLE else 0.0)


def order_direction(trade: Trade | None, fills: Sequence[Fill]) -> float:
    if trade is not None:
        return 1.0 if trade.order.action == "BUY" else -1.0
    return 1.0 if fills[0].execution.side == "BOT" else -1.0


def order_status(trade: Trade | None, filled_shares: float) -> str:
    if trade is None:
        return "filled"
    status = trade.orderStatus.status
    if status in STATUSES:
        return STATUSES[status]
    return "partially_filled" if filled_shares else "accepted"


def fill_commission(fills: Sequence[Fill]) -> float | None:
    if not fills or any(fill.commissionReport.execId != fill.execution.execId for fill in fills):
        return None
    return sum(float(fill.commissionReport.commission) for fill in fills)
