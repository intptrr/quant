import math

import numpy as np
import pytest

from quant.backtest.engine import Costs, Orders, Simulation, next_orders, simulate
from quant.core.trading import BrokerTerms

TIERED = BrokerTerms(
    commission_per_share=0.0035,
    commission_minimum=0.35,
    fees_per_share=0.0017,
    sale_fee_rate=0.0000206,
)


def column(values: list[float]) -> np.ndarray:
    return np.array(values).reshape(len(values), 1)


def orders(
    days: int,
    *,
    terms: BrokerTerms | None = None,
    min_value: float = 0.0,
    prices: list[float] | None = None,
) -> Orders:
    return Orders(column(prices or [100.0] * days), 100_000.0, terms or BrokerTerms(), min_value)


def run(
    targets: list[float],
    *,
    overnight: list[float] | None = None,
    intraday: list[float] | None = None,
    bands: list[float] | None = None,
    slippage: float = 0.0,
    borrow_fee: float = 0.0,
    debit: list[float] | None = None,
    credit: list[float] | None = None,
    credit_threshold: float = 0.0,
    terms: BrokerTerms | None = None,
    min_value: float = 0.0,
    prices: list[float] | None = None,
) -> Simulation:
    days = len(targets)
    zeros = [0.0] * days
    return simulate(
        column(overnight or zeros),
        column(intraday or zeros),
        column(targets),
        column(bands or zeros),
        Costs(
            slippage=slippage,
            borrow_fee=borrow_fee,
            debit_rate=np.array(debit or zeros),
            credit_rate=np.array(credit or zeros),
            credit_threshold=credit_threshold,
        ),
        orders(days, terms=terms, min_value=min_value, prices=prices),
    )


def test_no_targets_keep_equity_flat() -> None:
    simulation = run([0.0] * 5, overnight=[0.01] * 5, intraday=[0.02] * 5)

    assert simulation.equity.tolist() == pytest.approx([1.0] * 5)
    assert not simulation.traded.any()


def test_orders_decided_at_the_close_fill_at_the_next_open() -> None:
    simulation = run([1.0, 1.0, 1.0], overnight=[0.0, 0.1, 0.0], intraday=[0.0, 0.0, 0.05])

    assert simulation.traded[:, 0].tolist() == pytest.approx([0.0, 1.1, -0.1])
    assert simulation.equity.tolist() == pytest.approx([1.0, 1.0, 1.05])
    assert simulation.weights[1, 0] == pytest.approx(1.1)
    assert simulation.orders.tolist() == [0, 1, 1]


def test_full_investment_earns_the_asset_return_without_overnight_gaps() -> None:
    intraday = [0.0, 0.01, -0.02, 0.03]

    simulation = run([1.0] * 4, intraday=intraday)

    assert simulation.equity.tolist() == pytest.approx(np.cumprod(np.add(intraday, 1.0)).tolist())


def test_slippage_is_charged_on_executed_notional() -> None:
    simulation = run([1.0, 1.0], slippage=0.001)

    assert simulation.costs.tolist() == pytest.approx([0.0, 0.001])
    assert simulation.equity[-1] == pytest.approx(0.999)


def test_commissions_are_charged_per_order_in_dollars_of_the_account() -> None:
    simulation = run([1.0, 1.0], terms=TIERED)

    dollars = max(0.35, 0.0035 * 1000) + 0.0017 * 1000
    assert simulation.commissions.tolist() == pytest.approx([0.0, dollars / 100_000])


def test_small_orders_pay_the_minimum_commission_and_sales_pay_the_sec_fee() -> None:
    simulation = run([-0.001, -0.001], terms=TIERED)

    dollars = 0.35 + 0.0017 * 1 + 0.0000206 * 100
    assert simulation.commissions[1] == pytest.approx(dollars / 100_000)


def test_trades_stop_at_the_edge_of_the_band() -> None:
    simulation = run([1.0, 1.0, 0.5, 0.5], bands=[0.1] * 4)

    assert simulation.traded[:, 0].tolist() == pytest.approx([0.0, 0.9, 0.0, -0.3])
    assert simulation.weights[-1, 0] == pytest.approx(0.6)


def test_changes_inside_the_band_are_not_traded() -> None:
    simulation = run([1.0, 0.95, 0.85, 0.99, 0.99], bands=[0.1] * 5)

    assert np.count_nonzero(simulation.traded) == 1


def test_orders_below_the_minimum_value_are_skipped() -> None:
    simulation = run([0.5, 0.505, 0.505, 0.6, 0.6], min_value=1_000.0)

    assert np.count_nonzero(simulation.traded) == 2


@pytest.mark.parametrize(
    ("weight", "target", "change"), [(0.0, 0.005, 0.0), (0.005, 0.0, -0.005), (0.5, 0.52, 0.02)]
)
def test_small_orders_are_only_allowed_when_they_close_the_position(
    weight: float, target: float, change: float
) -> None:
    planned = next_orders(
        np.array([weight]), np.array([target]), np.array([0.0]), orders(1, min_value=1_000.0)
    )

    assert planned.tolist() == pytest.approx([change])


def test_a_long_position_is_closed_before_going_short() -> None:
    simulation = run([0.5, 0.5, -0.5, -0.5, -0.5])

    assert simulation.traded[:, 0].tolist() == pytest.approx([0.0, 0.5, 0.0, -0.5, -0.5])
    assert simulation.weights[:, 0].tolist() == pytest.approx([0.0, 0.5, 0.5, 0.0, -0.5])


def test_short_positions_pay_the_borrow_fee() -> None:
    simulation = run([-1.0, -1.0], borrow_fee=0.252)

    assert simulation.costs[-1] == pytest.approx(0.001)


def test_borrowed_cash_pays_the_debit_rate_of_the_day() -> None:
    simulation = run([2.0, 2.0, 2.0], debit=[0.0, 0.252, 0.504])

    assert simulation.costs.tolist() == pytest.approx([0.0, 0.001, 0.002], abs=1e-5)


def test_idle_cash_above_the_threshold_earns_the_credit_rate() -> None:
    simulation = run([0.5, 0.5], credit=[0.252, 0.252], credit_threshold=0.1)

    assert simulation.interest.tolist() == pytest.approx([0.0009, 0.0004], abs=1e-6)


def test_short_sale_proceeds_earn_no_interest() -> None:
    simulation = run([-0.5, -0.5], credit=[0.252, 0.252], credit_threshold=0.1)

    assert simulation.interest.tolist() == pytest.approx([0.0009, 0.0009], abs=1e-6)


def test_orders_are_dropped_on_days_without_a_price() -> None:
    simulation = run([1.0] * 8, prices=[100.0, math.nan, *[100.0] * 6])

    assert simulation.traded[:3, 0].tolist() == pytest.approx([0.0, 0.0, 1.0])
