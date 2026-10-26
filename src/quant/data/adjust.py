import polars as pl

from quant.data.schema import PRICE_COLUMNS


def adjust(bars: pl.DataFrame) -> pl.DataFrame:
    growth = (
        pl.col("split_factor") * (pl.col("close") + pl.col("dividend")) / pl.col("close").shift(1)
    )
    wealth = growth.fill_null(1.0).cum_prod()
    factor = wealth / wealth.last() * pl.col("close").last() / pl.col("close")
    return bars.with_columns(
        *[(pl.col(name) * factor).alias(f"adj_{name}") for name in PRICE_COLUMNS],
        (growth - 1.0).alias("total_return"),
    )
