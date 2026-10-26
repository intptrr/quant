import tomllib
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Protocol

import polars as pl
from pydantic import ConfigDict, TypeAdapter

from quant.strategies.fixed import FixedWeights


class Strategy(Protocol):
    @property
    def symbols(self) -> list[str]: ...

    def targets(self, prices: pl.DataFrame) -> pl.DataFrame: ...


TYPES = {
    "fixed": TypeAdapter(dict[str, FixedWeights], config=ConfigDict(title="fixed.toml")),
}


def load_strategies(folder: Path) -> dict[str, Strategy]:
    strategies: dict[str, Strategy] = {}
    for path in sorted(folder.glob("*.toml")):
        if path.stem not in TYPES:
            raise ValueError(f"{path} isn't named after a strategy type: {', '.join(TYPES)}")
        variants = TYPES[path.stem].validate_python(tomllib.loads(path.read_text()))
        if reused := sorted(set(variants) & set(strategies)):
            raise ValueError(f"{path} reuses the names {', '.join(reused)}")
        strategies |= variants
    return strategies


def select(strategies: Mapping[str, Strategy], patterns: Sequence[str]) -> dict[str, Strategy]:
    if not patterns:
        return dict(strategies)
    chosen: dict[str, Strategy] = {}
    for pattern in patterns:
        matches = [name for name in strategies if fnmatchcase(name, pattern)]
        if not matches:
            raise ValueError(f"no strategy matches {pattern}")
        chosen |= {name: strategies[name] for name in matches}
    return chosen
