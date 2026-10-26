from pathlib import Path

import polars as pl

from quant.data.schema import conform


class BarStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def path(self, provider: str, symbol: str) -> Path:
        return self._root / "raw" / provider / f"{symbol}.parquet"

    def read(self, provider: str, symbol: str) -> pl.DataFrame | None:
        path = self.path(provider, symbol)
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def write(self, provider: str, symbol: str, bars: pl.DataFrame) -> None:
        path = self.path(provider, symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(".parquet.tmp")
        conform(bars).write_parquet(staging)
        staging.replace(path)
