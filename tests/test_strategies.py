from pathlib import Path

import polars as pl
import pytest
from factories import make_prices, random_walk

from quant.strategies import TYPES, Strategy, load_strategies, select
from quant.strategies.fixed import FixedWeights

PROJECT_STRATEGIES = load_strategies(Path(__file__).parents[1] / "configs" / "strategies")
SPY = FixedWeights(weights={"SPY": 1.0})


def test_each_file_holds_named_variants_of_the_type_it_is_named_after(tmp_path: Path) -> None:
    (tmp_path / "fixed.toml").write_text(
        "[spy]\nweights = { SPY = 1.0 }\n\n[ief]\nweights = { IEF = 1.0 }\n"
    )

    strategies = load_strategies(tmp_path)

    assert strategies == {"spy": SPY, "ief": FixedWeights(weights={"IEF": 1.0})}


def test_files_must_be_named_after_a_strategy_type(tmp_path: Path) -> None:
    (tmp_path / "momentum.toml").write_text("[spy]\nweights = { SPY = 1.0 }\n")

    with pytest.raises(
        ValueError, match=r"momentum\.toml isn't named after a strategy type: fixed"
    ):
        load_strategies(tmp_path)


def test_parameter_errors_name_the_file_and_the_variant(tmp_path: Path) -> None:
    (tmp_path / "fixed.toml").write_text("[spy]\nweights = { SPY = 1.0 }\nbuffer = 0.1\n")

    with pytest.raises(ValueError, match=r"(?s)fixed\.toml.*spy\.buffer"):
        load_strategies(tmp_path)


def test_names_must_be_unique_across_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(TYPES, "copy", TYPES["fixed"])
    for name in ("fixed", "copy"):
        (tmp_path / f"{name}.toml").write_text("[spy]\nweights = { SPY = 1.0 }\n")

    with pytest.raises(ValueError, match="reuses the names spy"):
        load_strategies(tmp_path)


def test_names_and_patterns_select_strategies() -> None:
    strategies: dict[str, Strategy] = dict.fromkeys(
        ("spy", "sixty-forty", "sixty-forty-levered"), SPY
    )

    assert list(select(strategies, [])) == ["spy", "sixty-forty", "sixty-forty-levered"]
    assert list(select(strategies, ["sixty-*", "spy"])) == [
        "sixty-forty",
        "sixty-forty-levered",
        "spy",
    ]
    with pytest.raises(ValueError, match="no strategy matches gtaa"):
        select(strategies, ["gtaa"])


def test_the_project_has_the_spy_placeholder() -> None:
    assert PROJECT_STRATEGIES["spy"] == SPY


@pytest.mark.parametrize("name", list(PROJECT_STRATEGIES))
def test_targets_never_depend_on_later_prices(name: str) -> None:
    strategy = PROJECT_STRATEGIES[name]
    prices = make_prices(
        {symbol: random_walk(seed, 300) for seed, symbol in enumerate(strategy.symbols)}
    )
    full = strategy.targets(prices)

    for cut in prices["date"].unique().sort().gather([50, 150, 250]):
        before = pl.col("date") <= cut
        assert strategy.targets(prices.filter(before)).equals(full.filter(before))
