from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from quant.data.rates import RateStore, fetch_benchmark_rates, parse_fred_csv, rates_on

CSV = "observation_date,DFF\n2024-01-02,5.20\n2024-01-03,\n2024-01-04,5.18\n"
RATES = pl.DataFrame({"date": [date(2024, 1, 2), date(2024, 1, 4)], "rate": [0.052, 0.0518]})


def test_fred_csv_becomes_fractions_without_holidays() -> None:
    rates = parse_fred_csv(CSV)

    assert rates["date"].to_list() == RATES["date"].to_list()
    assert rates["rate"].to_list() == pytest.approx(RATES["rate"].to_list())


def test_rates_are_downloaded_from_fred_since_1997() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=CSV)

    rates = fetch_benchmark_rates(httpx.MockTransport(handler))

    (request,) = requests
    assert dict(request.url.params) == {"id": "DFF", "cosd": "1997-01-01"}
    assert rates.height == 2


def test_each_date_gets_the_latest_rate_known_on_that_day() -> None:
    days = [date(2024, 1, 5), date(2024, 1, 3), date(2024, 1, 2)]

    assert rates_on(days, RATES).tolist() == pytest.approx([0.0518, 0.052, 0.052])


def test_dates_before_the_first_rate_are_an_error() -> None:
    with pytest.raises(ValueError, match="no benchmark rate on or before 2024-01-01"):
        rates_on([date(2024, 1, 1)], RATES)


def test_rates_round_trip_through_the_store(tmp_path: Path) -> None:
    store = RateStore(tmp_path)
    with pytest.raises(ValueError, match="run `quant data sync`"):
        store.read()

    store.write(RATES)

    assert store.read().equals(RATES)
