import math

import numpy as np
import pytest

from quant.analytics import metrics

ALTERNATING = np.array([0.01, -0.005] * 126)


def test_sharpe_is_annualized_with_the_square_root_of_trading_days() -> None:
    expected = ALTERNATING.mean() / ALTERNATING.std(ddof=1) * math.sqrt(252)

    assert metrics.sharpe_ratio(ALTERNATING) == pytest.approx(expected)


def test_cagr_of_one_year_is_the_total_growth() -> None:
    daily = np.full(252, 1.1 ** (1 / 252) - 1.0)

    assert metrics.cagr(daily) == pytest.approx(0.1)


def test_annualized_volatility() -> None:
    expected = ALTERNATING.std(ddof=1) * math.sqrt(252)

    assert metrics.annualized_volatility(ALTERNATING) == pytest.approx(expected)


def test_max_drawdown_is_the_worst_fall_from_a_peak() -> None:
    assert metrics.max_drawdown([0.1, -0.5, 0.2, 0.5, 0.0]) == pytest.approx(-0.5)


def test_drawdown_counts_losses_from_the_starting_capital() -> None:
    assert metrics.max_drawdown([-0.1, 0.05]) == pytest.approx(-0.1)


def test_missing_values_are_ignored() -> None:
    assert metrics.sharpe_ratio([0.01, math.nan, -0.005, 0.002]) == pytest.approx(
        metrics.sharpe_ratio([0.01, -0.005, 0.002])
    )


def test_the_sharpe_ratio_of_flat_returns_is_nan() -> None:
    assert math.isnan(metrics.sharpe_ratio(np.zeros(10)))


def test_summary_reports_every_metric() -> None:
    summary = metrics.summarize(ALTERNATING, np.zeros(ALTERNATING.size))

    assert summary.cagr == pytest.approx(metrics.cagr(ALTERNATING))
    assert summary.volatility == pytest.approx(metrics.annualized_volatility(ALTERNATING))
    assert summary.sharpe == pytest.approx(metrics.sharpe_ratio(ALTERNATING))
    assert summary.max_drawdown == pytest.approx(metrics.max_drawdown(ALTERNATING))


def test_the_summary_sharpe_ratio_is_measured_above_the_risk_free_rate() -> None:
    summary = metrics.summarize(ALTERNATING, np.full(ALTERNATING.size, 0.001))

    assert summary.sharpe == pytest.approx(metrics.sharpe_ratio(ALTERNATING - 0.001))
    assert summary.sharpe < metrics.sharpe_ratio(ALTERNATING)
    assert summary.cagr == pytest.approx(metrics.cagr(ALTERNATING))
