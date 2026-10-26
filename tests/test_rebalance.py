import numpy as np
import pytest

from quant.core.rebalance import desired_weights


def desired(current: float, target: float, band: float) -> float:
    return float(desired_weights(np.array([current]), np.array([target]), np.array([band]))[0])


def test_positions_inside_the_band_are_kept() -> None:
    assert desired(0.95, 1.0, 0.1) == pytest.approx(0.95)


def test_positions_outside_the_band_move_to_its_nearest_edge() -> None:
    assert desired(0.5, 1.0, 0.1) == pytest.approx(0.9)
    assert desired(1.5, 1.0, 0.1) == pytest.approx(1.1)


def test_a_change_of_side_closes_the_position_first() -> None:
    assert desired(0.5, -1.0, 0.1) == 0.0
    assert desired(-0.5, 1.0, 0.1) == 0.0


def test_a_flat_position_opens_on_either_side() -> None:
    assert desired(0.0, -1.0, 0.1) == pytest.approx(-0.9)
