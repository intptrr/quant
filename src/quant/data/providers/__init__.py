from datetime import date
from typing import Protocol

import polars as pl

from quant.data.providers.sharadar import SharadarProvider
from quant.settings import Settings

SHARADAR = "sharadar"


class PriceProvider(Protocol):
    @property
    def name(self) -> str: ...

    def fetch_daily(self, symbol: str, start: date, end: date) -> pl.DataFrame: ...


def sharadar_provider(settings: Settings) -> SharadarProvider:
    if settings.sharadar_api_key is None:
        raise ValueError("SHARADAR_API_KEY is not set; add it to .env")
    return SharadarProvider(settings.sharadar_api_key.get_secret_value())
