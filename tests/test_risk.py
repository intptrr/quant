import pytest
from factories import trading_with

from quant.live.planner import OrderIntent
from quant.live.risk import Rejection, screen_orders

LIMITS = trading_with(max_order_fraction=0.25)


def screen(
    intent: OrderIntent,
    *,
    latest: float | None = None,
    shortable: bool = True,
    split: str | None = None,
    drawdown: float = 0.0,
) -> list[Rejection]:
    approved, rejected = screen_orders(
        [intent],
        equity=100_000.0,
        latest_prices={} if latest is None else {intent.symbol: latest},
        shortable={intent.symbol: shortable},
        splits={} if split is None else {intent.symbol: split},
        drawdown=drawdown,
        trading=LIMITS,
    )
    assert len(approved) + len(rejected) == 1
    return rejected


def reasons(
    intent: OrderIntent,
    *,
    latest: float | None = None,
    shortable: bool = True,
    drawdown: float = 0.0,
) -> list[str]:
    rejected = screen(intent, latest=latest, shortable=shortable, drawdown=drawdown)
    return [rejection.reason for rejection in rejected]


def test_ordinary_orders_are_approved() -> None:
    assert reasons(OrderIntent("SPY", 10, 500.0, 0.0), latest=505.0) == []


def test_orders_larger_than_the_limit_are_rejected() -> None:
    (reason,) = reasons(OrderIntent("SPY", 60, 500.0, 0.0))

    assert reason.startswith("order value 30,000 exceeds 25% of equity")


def test_prices_far_from_the_last_close_block_trading() -> None:
    (reason,) = reasons(OrderIntent("USO", 10, 2.13, 0.0), latest=18.0)

    assert "far from last close" in reason


def test_symbols_splitting_at_the_open_wait_for_the_next_session() -> None:
    (rejection,) = screen(OrderIntent("USO", 100, 2.13, 0.0), latest=2.13, split="1-for-8 split")

    assert rejection.reason == "1-for-8 split takes effect at this open, trading it next session"
    assert rejection.needs_attention


@pytest.mark.parametrize(
    ("intent", "blocked"),
    [
        (OrderIntent("SPY", 10, 500.0, 10.0), True),
        (OrderIntent("SPY", -10, 500.0, 20.0), False),
    ],
)
def test_kill_switch_only_allows_reducing_positions(intent: OrderIntent, blocked: bool) -> None:
    assert bool(reasons(intent, drawdown=-0.3)) == blocked


def test_shorts_need_an_easy_to_borrow_symbol_without_raising_an_alarm() -> None:
    (rejection,) = screen(OrderIntent("FXY", -10, 60.0, 0.0), shortable=False)

    assert rejection.reason == "not easy to borrow, staying out of the short"
    assert not rejection.needs_attention


def test_other_rejections_need_attention() -> None:
    (rejection,) = screen(OrderIntent("SPY", 60, 500.0, 0.0))

    assert rejection.needs_attention


def test_covering_a_short_never_needs_a_borrow() -> None:
    assert reasons(OrderIntent("FXY", 10, 60.0, -20.0), shortable=False) == []
