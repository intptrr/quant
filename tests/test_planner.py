from quant.live.planner import OrderIntent, Target, below_minimum, plan_orders

PRICES = {"SPY": 500.0, "TLT": 100.0, "GLD": 200.0}


def quantities(intents: list[OrderIntent]) -> dict[str, int]:
    return {intent.symbol: intent.quantity for intent in intents}


def test_targets_become_whole_share_orders() -> None:
    targets = {"SPY": Target(0.5, 0.0), "TLT": Target(-0.25, 0.0)}

    intents = plan_orders(targets, PRICES, {}, 100_000.0)

    assert quantities(intents) == {"SPY": 100, "TLT": -250}


def test_whole_shares_never_take_a_position_past_its_target() -> None:
    targets = {"SPY": Target(1.0, 0.0), "TLT": Target(-1.0, 0.0)}

    intents = plan_orders(targets, {"SPY": 600.0, "TLT": 600.0}, {}, 100_000.0)

    assert quantities(intents) == {"SPY": 166, "TLT": -166}


def test_positions_inside_the_band_are_not_traded() -> None:
    targets = {"SPY": Target(0.5, 0.1)}

    assert plan_orders(targets, PRICES, {"SPY": 90.0}, 100_000.0) == []


def test_positions_outside_the_band_trade_to_its_edge() -> None:
    targets = {"SPY": Target(0.5, 0.1)}

    assert quantities(plan_orders(targets, PRICES, {"SPY": 50.0}, 100_000.0)) == {"SPY": 30}


def test_switching_sides_only_closes_the_position() -> None:
    targets = {"TLT": Target(-0.5, 0.0)}

    assert quantities(plan_orders(targets, PRICES, {"TLT": 120.0}, 100_000.0)) == {"TLT": -120}


def test_held_symbols_without_a_target_are_closed() -> None:
    intents = plan_orders({}, PRICES, {"GLD": 15.0}, 100_000.0)

    assert quantities(intents) == {"GLD": -15}
    assert intents[0].current == 15.0
    assert not intents[0].adds_risk


def test_order_intents_know_whether_they_add_risk() -> None:
    assert OrderIntent("SPY", 10, 500.0, 5.0).adds_risk
    assert OrderIntent("SPY", -10, 500.0, 0.0).adds_risk
    assert not OrderIntent("SPY", -5, 500.0, 10.0).adds_risk


def test_orders_below_the_minimum_value_are_flagged() -> None:
    assert below_minimum(OrderIntent("TLT", 5, 100.0, 0.0), 1_000.0)
    assert not below_minimum(OrderIntent("SPY", 2, 500.0, 0.0), 1_000.0)


def test_small_orders_that_close_a_position_are_never_below_the_minimum() -> None:
    assert not below_minimum(OrderIntent("TLT", -3, 100.0, 3.0), 1_000.0)
