from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant.analytics.metrics import TRADING_DAYS
from quant.core.rebalance import desired_weights
from quant.core.trading import BrokerTerms

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Costs:
    slippage: float
    borrow_fee: float
    debit_rate: FloatArray
    credit_rate: FloatArray
    credit_threshold: float


@dataclass(frozen=True, slots=True)
class Orders:
    prices: FloatArray
    capital: float
    terms: BrokerTerms
    min_value: float


@dataclass(frozen=True, slots=True)
class Simulation:
    equity: FloatArray
    weights: FloatArray
    traded: FloatArray
    slippage: FloatArray
    commissions: FloatArray
    borrow: FloatArray
    financing: FloatArray
    interest: FloatArray
    orders: npt.NDArray[np.int64]

    @property
    def costs(self) -> FloatArray:
        return self.slippage + self.commissions + self.borrow + self.financing


def simulate(
    overnight: FloatArray,
    intraday: FloatArray,
    targets: FloatArray,
    bands: FloatArray,
    costs: Costs,
    orders: Orders,
) -> Simulation:
    overnight, intraday = np.nan_to_num(overnight), np.nan_to_num(intraday)
    targets, bands = np.nan_to_num(targets), np.nan_to_num(bands)
    tradable = np.isfinite(orders.prices) & (orders.prices > 0.0)
    days, count = overnight.shape
    equity = np.empty(days)
    weights = np.empty((days, count))
    traded = np.zeros((days, count))
    slippage, commissions, borrow = np.zeros(days), np.zeros(days), np.zeros(days)
    financing, interest = np.zeros(days), np.zeros(days)
    order_counts = np.zeros(days, dtype=np.int64)
    cash = 1.0
    held, pending = np.zeros(count), np.zeros(count)
    for day in range(days):
        start = equity[day - 1] if day else 1.0
        opened = np.where(tradable[day], pending * (1.0 + overnight[day]), 0.0)
        held = held * (1.0 + overnight[day]) + opened
        cash -= float(opened.sum())
        short = float(-np.minimum(held, 0.0).sum())
        idle = cash - short - costs.credit_threshold * start
        slippage[day] = costs.slippage * float(np.abs(opened).sum())
        commissions[day] = commission_cost(opened, start, orders.prices[day], orders)
        borrow[day] = costs.borrow_fee * short / TRADING_DAYS
        financing[day] = costs.debit_rate[day] * max(-cash, 0.0) / TRADING_DAYS
        interest[day] = costs.credit_rate[day] * max(idle, 0.0) / TRADING_DAYS
        held = held * (1.0 + intraday[day])
        cash += interest[day] - slippage[day] - commissions[day] - borrow[day] - financing[day]
        equity[day] = cash + float(held.sum())
        weights[day] = held / equity[day]
        traded[day] = opened
        order_counts[day] = np.count_nonzero(opened)
        pending = next_orders(weights[day], targets[day], bands[day], orders) * equity[day]
    return Simulation(
        equity, weights, traded, slippage, commissions, borrow, financing, interest, order_counts
    )


def next_orders(
    weights: FloatArray, targets: FloatArray, bands: FloatArray, orders: Orders
) -> FloatArray:
    desired = desired_weights(weights, targets, bands)
    change = desired - weights
    too_small = (np.abs(change) * orders.capital < orders.min_value) & (desired != 0.0)
    return np.where(too_small, 0.0, change)


def commission_cost(
    executed: FloatArray, equity: float, prices: FloatArray, orders: Orders
) -> float:
    traded = executed != 0.0
    if not traded.any():
        return 0.0
    value = executed[traded] / equity * orders.capital
    quantities = value / prices[traded]
    dollars = orders.terms.commissions(quantities, prices[traded])
    return float(dollars.sum()) / orders.capital * equity
