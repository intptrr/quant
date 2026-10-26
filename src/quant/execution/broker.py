from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

TERMINAL_STATUSES = frozenset(
    {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"}
)


class OrderRejectedError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    client_order_id: str
    symbol: str
    quantity: float
    filled_quantity: float
    filled_price: float | None
    status: str
    commission: float | None = None

    @property
    def done(self) -> bool:
        return self.status in TERMINAL_STATUSES


class Broker(Protocol):
    def equity(self) -> float: ...

    def positions(self) -> dict[str, float]: ...

    def open_orders(self) -> list[BrokerOrder]: ...

    def order(self, client_order_id: str) -> BrokerOrder | None: ...

    def submit_opening_order(
        self, client_order_id: str, symbol: str, quantity: int
    ) -> BrokerOrder: ...

    def shortable(self, symbols: Sequence[str]) -> dict[str, bool]: ...

    def close(self) -> None: ...


class MarketWatch(Protocol):
    def latest_prices(self, symbols: Sequence[str]) -> dict[str, float]: ...

    def splits(self, symbols: Sequence[str], start: date, end: date) -> dict[str, str]: ...
