from collections.abc import Sequence

import polars as pl

from quant.data.adjust import adjust
from quant.data.store import BarStore


def load_adjusted(store: BarStore, provider: str, symbols: Sequence[str]) -> pl.DataFrame:
    frames = []
    for symbol in symbols:
        bars = store.read(provider, symbol)
        if bars is None:
            raise FileNotFoundError(f"no {provider} bars for {symbol}; run `quant data sync` first")
        frames.append(adjust(bars).with_columns(pl.lit(symbol).alias("symbol")))
    return pl.concat(frames).select("date", "symbol", pl.exclude("date", "symbol"))
