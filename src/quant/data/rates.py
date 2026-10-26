import io
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import numpy.typing as npt
import polars as pl

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BENCHMARK_SERIES = "DFF"
HISTORY_START = date(1997, 1, 1)
PERCENT = 100.0


class RateStore:
    def __init__(self, root: Path) -> None:
        self._path = root / "raw" / "fred" / f"{BENCHMARK_SERIES}.parquet"

    def read(self) -> pl.DataFrame:
        if not self._path.exists():
            raise ValueError("no benchmark rates stored; run `quant data sync`")
        return pl.read_parquet(self._path)

    def write(self, rates: pl.DataFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        staging = self._path.with_suffix(".parquet.tmp")
        rates.write_parquet(staging)
        staging.replace(self._path)


def fetch_benchmark_rates(transport: httpx.BaseTransport | None = None) -> pl.DataFrame:
    with httpx.Client(timeout=30.0, transport=transport) as client:
        response = client.get(
            FRED_CSV_URL, params={"id": BENCHMARK_SERIES, "cosd": HISTORY_START.isoformat()}
        )
    response.raise_for_status()
    return parse_fred_csv(response.text)


def parse_fred_csv(text: str) -> pl.DataFrame:
    frame = pl.read_csv(
        io.StringIO(text), schema={"observation_date": pl.Date(), BENCHMARK_SERIES: pl.Float64()}
    )
    return frame.select(
        pl.col("observation_date").alias("date"),
        (pl.col(BENCHMARK_SERIES) / PERCENT).alias("rate"),
    ).drop_nulls()


def rates_on(dates: Sequence[date] | pl.Series, rates: pl.DataFrame) -> npt.NDArray[np.float64]:
    wanted = pl.DataFrame({"date": dates}, schema={"date": pl.Date()}).with_row_index("order")
    joined = (
        wanted.sort("date")
        .join_asof(rates.sort("date"), on="date", strategy="backward")
        .sort("order")
    )
    missing = joined.filter(pl.col("rate").is_null())
    if not missing.is_empty():
        raise ValueError(f"no benchmark rate on or before {missing['date'][0]}")
    return joined["rate"].to_numpy()
