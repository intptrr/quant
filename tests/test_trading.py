from pathlib import Path

import numpy as np
import pytest

from quant.core.trading import BrokerName, BrokerTerms, TradingConfig, load_trading_config

PROJECT = Path(__file__).parents[1]

TIERED = BrokerTerms(
    margin_spread=0.015,
    cash_spread=0.005,
    cash_threshold=10_000.0,
    full_rate_nav=100_000.0,
    commission_per_share=0.0035,
    commission_minimum=0.35,
    commission_maximum=0.01,
    fees_per_share=0.0017,
    sale_fee_rate=0.0000206,
)


def commission(quantity: float, price: float, terms: BrokerTerms = TIERED) -> float:
    return float(terms.commissions(np.array([quantity]), np.array([price]))[0])


def test_commissions_follow_the_per_share_rate_with_a_minimum_and_a_cap() -> None:
    assert commission(1_000, 50.0) == pytest.approx(3.5 + 1.7)
    assert commission(10, 50.0) == pytest.approx(0.35 + 0.017)
    assert commission(10, 2.0) == pytest.approx(0.2 + 0.017)


def test_sales_also_pay_the_sec_fee() -> None:
    assert commission(-1_000, 50.0) == pytest.approx(3.5 + 1.7 + 0.0000206 * 50_000)


def test_a_broker_without_commissions_charges_nothing() -> None:
    assert commission(1_000, 50.0, BrokerTerms()) == 0.0


def test_credit_interest_needs_a_cash_spread_and_shrinks_below_the_full_rate_nav() -> None:
    benchmark = np.array([0.04, 0.003])

    assert TIERED.debit_rate(benchmark).tolist() == pytest.approx([0.055, 0.018])
    assert TIERED.credit_rate(benchmark, 100_000.0).tolist() == pytest.approx([0.035, 0.0])
    assert TIERED.credit_rate(benchmark, 50_000.0).tolist() == pytest.approx([0.0175, 0.0])
    assert BrokerTerms().credit_rate(benchmark, 100_000.0).tolist() == [0.0, 0.0]


def test_the_selected_broker_must_have_terms() -> None:
    with pytest.raises(ValueError, match=r"no \[brokers.ibkr\] terms"):
        TradingConfig(brokers={BrokerName.ALPACA: BrokerTerms()})


def test_switching_brokers_keeps_everything_else() -> None:
    trading = TradingConfig(
        min_order_value=1_000.0,
        brokers={BrokerName.IBKR: TIERED, BrokerName.ALPACA: BrokerTerms(margin_spread=0.026)},
    )

    alpaca = trading.using(BrokerName.ALPACA)

    assert alpaca.terms.margin_spread == 0.026
    assert alpaca.min_order_value == 1_000.0
    with pytest.raises(ValueError, match=r"no \[brokers.alpaca\] terms"):
        TradingConfig(brokers={BrokerName.IBKR: TIERED}).using(BrokerName.ALPACA)


def test_the_project_trading_config_loads_both_brokers() -> None:
    trading = load_trading_config(PROJECT / "configs" / "trading.toml")

    assert trading.broker is BrokerName.IBKR
    assert trading.strategy == "spy"
    assert trading.min_order_value == 1_000.0
    assert trading.terms.margin_spread == 0.015
    assert trading.using(BrokerName.ALPACA).terms.margin_spread == 0.026


def test_unknown_config_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="extra_forbidden"):
        TradingConfig.model_validate({"slippage": {"default_bps": 3.0}, "brokers": {"ibkr": {}}})
