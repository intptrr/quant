from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from quant.core.rebalance import desired_weights


@dataclass(frozen=True, slots=True)
class Target:
    weight: float
    band: float


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    quantity: int
    price: float
    current: float

    @property
    def resulting(self) -> float:
        return self.current + self.quantity

    @property
    def adds_risk(self) -> bool:
        return abs(self.resulting) > abs(self.current)


def plan_orders(
    targets: Mapping[str, Target],
    prices: Mapping[str, float],
    positions: Mapping[str, float],
    equity: float,
) -> list[OrderIntent]:
    symbols = sorted(set(targets) | set(positions))
    flat = Target(0.0, 0.0)
    shares = np.array([positions.get(symbol, 0.0) for symbol in symbols])
    price = np.array([prices[symbol] for symbol in symbols])
    desired = desired_weights(
        shares * price / equity,
        np.array([targets.get(symbol, flat).weight for symbol in symbols]),
        np.array([targets.get(symbol, flat).band for symbol in symbols]),
    )
    quantities = np.round(np.trunc(desired * equity / price) - shares)
    return [
        OrderIntent(symbol, int(quantity), float(close), float(current))
        for symbol, quantity, close, current in zip(symbols, quantities, price, shares, strict=True)
        if quantity != 0
    ]


def below_minimum(intent: OrderIntent, min_order_value: float) -> bool:
    return abs(intent.quantity) * intent.price < min_order_value and intent.resulting != 0
