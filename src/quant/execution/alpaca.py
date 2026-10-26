from collections.abc import Sequence
from datetime import date

from alpaca.common.exceptions import APIError
from alpaca.data.enums import CorporateActionsType, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.models.corporate_actions import (
    CorporateActionsSet,
    ForwardSplit,
    ReverseSplit,
    StockDividend,
)
from alpaca.data.requests import CorporateActionsRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Asset, Order, Position, TradeAccount
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from quant.execution.broker import BrokerOrder, OrderRejectedError

NOT_FOUND = 404
REFUSED = frozenset({403, 422})
SPLIT_TYPES = [
    CorporateActionsType.FORWARD_SPLIT,
    CorporateActionsType.REVERSE_SPLIT,
    CorporateActionsType.STOCK_DIVIDEND,
]


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str) -> None:
        self._trading = TradingClient(api_key, secret_key, paper=True)

    def equity(self) -> float:
        return float(expect(self._trading.get_account(), TradeAccount).equity or 0.0)

    def positions(self) -> dict[str, float]:
        return {
            position.symbol: signed_quantity(position)
            for position in expect(self._trading.get_all_positions(), list)
            if isinstance(position, Position)
        }

    def open_orders(self) -> list[BrokerOrder]:
        orders = self._trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        return [
            to_broker_order(order) for order in expect(orders, list) if isinstance(order, Order)
        ]

    def order(self, client_order_id: str) -> BrokerOrder | None:
        try:
            order = self._trading.get_order_by_client_id(client_order_id)
        except APIError as error:
            if error.status_code == NOT_FOUND:
                return None
            raise
        return to_broker_order(expect(order, Order))

    def submit_opening_order(self, client_order_id: str, symbol: str, quantity: int) -> BrokerOrder:
        request = MarketOrderRequest(
            symbol=symbol,
            qty=abs(quantity),
            side=OrderSide.BUY if quantity > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.OPG,
            client_order_id=client_order_id,
        )
        try:
            order = self._trading.submit_order(request)
        except APIError as error:
            if error.status_code not in REFUSED:
                raise
            raise OrderRejectedError(error.message) from error
        return to_broker_order(expect(order, Order))

    def shortable(self, symbols: Sequence[str]) -> dict[str, bool]:
        assets = {symbol: expect(self._trading.get_asset(symbol), Asset) for symbol in symbols}
        return {
            symbol: bool(asset.shortable and asset.easy_to_borrow)
            for symbol, asset in assets.items()
        }

    def close(self) -> None:
        return None


class AlpacaMarketWatch:
    def __init__(self, api_key: str, secret_key: str) -> None:
        self._data = StockHistoricalDataClient(api_key, secret_key)
        self._actions = CorporateActionsClient(api_key, secret_key)

    def latest_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        if not symbols:
            return {}
        trades = self._data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX)
        )
        return {symbol: float(trade.price) for symbol, trade in expect(trades, dict).items()}

    def splits(self, symbols: Sequence[str], start: date, end: date) -> dict[str, str]:
        if not symbols:
            return {}
        actions = self._actions.get_corporate_actions(
            CorporateActionsRequest(symbols=list(symbols), types=SPLIT_TYPES, start=start, end=end)
        )
        return describe_splits(expect(actions, CorporateActionsSet), start, end)


def describe_splits(actions: CorporateActionsSet, start: date, end: date) -> dict[str, str]:
    return {
        action.symbol: describe_split(action)
        for group in actions.data.values()
        for action in group
        if isinstance(action, ForwardSplit | ReverseSplit | StockDividend)
        and start <= action.ex_date <= end
    }


def describe_split(action: ForwardSplit | ReverseSplit | StockDividend) -> str:
    if isinstance(action, StockDividend):
        return f"stock dividend of {action.rate:g} shares per share"
    return f"{action.new_rate:g}-for-{action.old_rate:g} split"


def signed_quantity(position: Position) -> float:
    quantity = abs(float(position.qty))
    return -quantity if position.side == PositionSide.SHORT else quantity


def to_broker_order(order: Order) -> BrokerOrder:
    direction = 1.0 if order.side == OrderSide.BUY else -1.0
    return BrokerOrder(
        client_order_id=order.client_order_id,
        symbol=order.symbol or "",
        quantity=direction * float(order.qty or 0.0),
        filled_quantity=direction * float(order.filled_qty or 0.0),
        filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
        status=str(order.status.value),
    )


def expect[T](value: object, kind: type[T]) -> T:
    if not isinstance(value, kind):
        raise TypeError(f"expected {kind.__name__} from Alpaca, got {type(value).__name__}")
    return value
