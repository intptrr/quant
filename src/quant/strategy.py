import tomllib
from pathlib import Path
from typing import Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class Strategy(Protocol):
    @property
    def symbols(self) -> list[str]: ...

    def targets(self, prices: pl.DataFrame) -> pl.DataFrame: ...


class FixedWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    weights: dict[str, float] = Field(min_length=1)
    rebalance_band: float = Field(default=0.0, ge=0)

    @property
    def symbols(self) -> list[str]:
        return list(self.weights)

    def targets(self, prices: pl.DataFrame) -> pl.DataFrame:
        weights = pl.DataFrame({"symbol": self.symbols, "weight": list(self.weights.values())})
        return (
            prices.select("date", "symbol")
            .join(weights, on="symbol")
            .with_columns((pl.col("weight").abs() * self.rebalance_band).alias("band"))
            .sort("date", "symbol")
        )


def load_strategy(path: Path) -> FixedWeights:
    return FixedWeights.model_validate(tomllib.loads(path.read_text()))
