import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

TRADING_DAYS = 252

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float


def summarize(returns: npt.ArrayLike) -> PerformanceSummary:
    values = clean(returns)
    return PerformanceSummary(
        cagr=cagr(values),
        volatility=annualized_volatility(values),
        sharpe=sharpe_ratio(values),
        max_drawdown=max_drawdown(values),
    )


def clean(returns: npt.ArrayLike) -> FloatArray:
    values = np.asarray(returns, dtype=np.float64)
    return values[~np.isnan(values)]


def cagr(returns: npt.ArrayLike) -> float:
    values = clean(returns)
    return float(np.prod(1.0 + values) ** (TRADING_DAYS / values.size) - 1.0)


def annualized_volatility(returns: npt.ArrayLike) -> float:
    return float(np.std(clean(returns), ddof=1)) * math.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: npt.ArrayLike) -> float:
    values = clean(returns)
    deviation = float(values.std(ddof=1))
    if not deviation:
        return math.nan
    return float(values.mean()) / deviation * math.sqrt(TRADING_DAYS)


def max_drawdown(returns: npt.ArrayLike) -> float:
    wealth = np.cumprod(1.0 + clean(returns))
    return float((wealth / np.maximum.accumulate(np.maximum(wealth, 1.0)) - 1.0).min())
