from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quant.core.trading import TradingConfig
from quant.live.planner import OrderIntent

NOT_BORROWABLE = "not easy to borrow, staying out of the short"


@dataclass(frozen=True, slots=True)
class Rejection:
    intent: OrderIntent
    reason: str

    @property
    def needs_attention(self) -> bool:
        return self.reason != NOT_BORROWABLE


def screen_orders(
    intents: Sequence[OrderIntent],
    *,
    equity: float,
    latest_prices: Mapping[str, float],
    shortable: Mapping[str, bool],
    splits: Mapping[str, str],
    drawdown: float,
    trading: TradingConfig,
) -> tuple[list[OrderIntent], list[Rejection]]:
    approved: list[OrderIntent] = []
    rejected: list[Rejection] = []
    for intent in intents:
        reason = rejection_reason(
            intent, equity, latest_prices, shortable, splits, drawdown, trading
        )
        if reason is None:
            approved.append(intent)
        else:
            rejected.append(Rejection(intent, reason))
    return approved, rejected


def rejection_reason(
    intent: OrderIntent,
    equity: float,
    latest_prices: Mapping[str, float],
    shortable: Mapping[str, bool],
    splits: Mapping[str, str],
    drawdown: float,
    trading: TradingConfig,
) -> str | None:
    if split := splits.get(intent.symbol):
        return f"{split} takes effect at this open, trading it next session"
    value = abs(intent.quantity) * intent.price
    if value > trading.max_order_fraction * equity:
        return f"order value {value:,.0f} exceeds {trading.max_order_fraction:.0%} of equity"
    latest = latest_prices.get(intent.symbol)
    if latest is not None and abs(latest / intent.price - 1.0) > trading.max_price_deviation:
        return f"latest price {latest:,.2f} is far from last close {intent.price:,.2f}"
    if intent.adds_risk and drawdown <= -trading.max_drawdown:
        return f"kill switch: drawdown {drawdown:.1%} only allows reducing positions"
    if intent.resulting < 0.0 and intent.adds_risk and not shortable.get(intent.symbol, False):
        return NOT_BORROWABLE
    return None
