import polars as pl

BAR_SCHEMA = pl.Schema(
    {
        "date": pl.Date(),
        "open": pl.Float64(),
        "high": pl.Float64(),
        "low": pl.Float64(),
        "close": pl.Float64(),
        "volume": pl.Float64(),
        "dividend": pl.Float64(),
        "split_factor": pl.Float64(),
    }
)

PRICE_COLUMNS = ("open", "high", "low", "close")


def conform(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(pl.col(name).cast(dtype) for name, dtype in BAR_SCHEMA.items()).sort("date")


def empty_bars() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)
