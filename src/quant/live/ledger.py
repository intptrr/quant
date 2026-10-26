from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from quant.analytics.metrics import TRADING_DAYS

FLAT = 1e-9


@dataclass(frozen=True, slots=True)
class Bar:
    open: float
    close: float
    dividend: float
    split_factor: float


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    quantity: float
    price: float
    commission: float = 0.0


@dataclass(frozen=True, slots=True)
class Carry:
    borrow_fee: float = 0.0
    debit_rate: float = 0.0
    credit_rate: float = 0.0
    credit_threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class LedgerDay:
    date: date
    cash: float
    positions: dict[str, float]
    equity: float
    peak_equity: float

    @property
    def drawdown(self) -> float:
        return self.equity / self.peak_equity - 1.0


def opening_day(day: date, capital: float) -> LedgerDay:
    return LedgerDay(day, capital, {}, capital, capital)


def close_day(
    previous: LedgerDay,
    day: date,
    bars: Mapping[str, Bar],
    fills: Sequence[Fill],
    carry: Carry,
) -> LedgerDay:
    cash = previous.cash
    positions: dict[str, float] = {}
    for symbol, shares in previous.positions.items():
        cash += shares * bars[symbol].dividend
        positions[symbol] = shares * bars[symbol].split_factor
    for fill in fills:
        cash -= fill.quantity * fill.price + fill.commission
        positions[fill.symbol] = positions.get(fill.symbol, 0.0) + fill.quantity
    positions = {symbol: shares for symbol, shares in positions.items() if abs(shares) > FLAT}
    values = {symbol: shares * bars[symbol].close for symbol, shares in positions.items()}
    short_value = -sum(value for value in values.values() if value < 0.0)
    idle = cash - short_value - carry.credit_threshold
    cash -= (
        carry.borrow_fee * short_value
        + carry.debit_rate * max(-cash, 0.0)
        - carry.credit_rate * max(idle, 0.0)
    ) / TRADING_DAYS
    equity = cash + sum(values.values())
    return LedgerDay(day, cash, positions, equity, max(previous.peak_equity, equity))
