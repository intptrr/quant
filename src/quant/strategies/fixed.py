import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class FixedWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    weights: dict[str, float] = Field(min_length=1)
    rebalance_band: float = Field(default=0.0, ge=0)

    @property
    def symbols(self) -> list[str]:
        return list(self.weights)

    def targets(self, prices: pl.DataFrame) -> pl.DataFrame:
        weights = pl.DataFrame({"symbol": self.symbols, "weight": list(self.weights.values())})
        held = prices.select("date", "symbol").join(weights, on="symbol")
        start = pl.col("date").min().over("symbol").max()
        return (
            held.filter(pl.col("date") >= start)
            .with_columns((pl.col("weight").abs() * self.rebalance_band).alias("band"))
            .sort("date", "symbol")
        )
