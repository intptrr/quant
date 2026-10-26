from datetime import date
from typing import Any

import httpx
import pytest
from fakes import KEY, FakeSharadar

from quant.data.providers.sharadar import PRICE_FIELDS, SharadarError, SharadarProvider
from quant.data.schema import BAR_SCHEMA

USO_LISTING = [{"ticker": "USO", "table": "funds", "firstpricedate": "2020-04-27"}]
USO_SPLIT_ADJUSTED_PRICES = [
    {
        "date": "2020-04-27",
        "open": 18.4,
        "high": 18.4,
        "low": 17.04,
        "close": 17.52,
        "volume": 33885000,
        "closeunadj": 2.19,
    },
    {
        "date": "2020-04-28",
        "open": 17.28,
        "high": 18.0,
        "low": 16.88,
        "close": 17.04,
        "volume": 19476850,
        "closeunadj": 2.13,
    },
    {
        "date": "2020-04-29",
        "open": 18.01,
        "high": 18.22,
        "low": 17.46,
        "close": 18,
        "volume": 23378600,
        "closeunadj": 18,
    },
]
USO_ACTIONS = [
    {"date": "2020-04-29", "action": "split", "value": 0.125},
    {"date": "2020-05-01", "action": "tickerchangeto", "value": None},
]


def uso(**tables: list[dict[str, Any]]) -> FakeSharadar:
    defaults = {"tickers": USO_LISTING, "funds": USO_SPLIT_ADJUSTED_PRICES, "actions": USO_ACTIONS}
    return FakeSharadar({**defaults, **tables})


def test_listing_prices_and_actions_are_requested() -> None:
    server = uso()

    server.provider().fetch_daily("USO", date(2020, 4, 27), date(2020, 4, 29))

    assert [request.url.path for request in server.requests] == [
        "/v1.0/data/tickers",
        "/v1.0/data/funds",
        "/v1.0/data/actions",
    ]
    params = server.requests[1].url.params
    assert params["api_key"] == KEY
    assert params["ticker"] == "USO"
    assert (params["from"], params["to"]) == ("2020-04-27", "2020-04-29")
    assert params["fields"] == PRICE_FIELDS
    assert (params["sort"], params["format"]) == ("date.asc", "json")


def test_reverse_split_adjustment_is_undone_like_uso_april_2020() -> None:
    bars = uso().provider().fetch_daily("USO", date(2020, 4, 27), date(2020, 4, 29))

    assert bars.schema == BAR_SCHEMA
    assert bars["date"].to_list() == [date(2020, 4, 27), date(2020, 4, 28), date(2020, 4, 29)]
    assert bars["close"].to_list() == pytest.approx([2.19, 2.13, 18.0])
    assert bars["open"].to_list() == pytest.approx([2.3, 2.16, 18.01])
    assert bars["volume"].to_list() == pytest.approx([271_080_000, 155_814_800, 23_378_600])
    assert bars["split_factor"].to_list() == [1.0, 1.0, 0.125]
    assert bars["dividend"].to_list() == [0.0, 0.0, 0.0]


def test_split_adjusted_dividends_are_restored_to_raw_amounts() -> None:
    server = FakeSharadar(
        {
            "tickers": [{"ticker": "AAPL", "table": "stocks", "firstpricedate": "2020-08-07"}],
            "stocks": [
                {
                    "date": "2020-08-07",
                    "open": 111.0,
                    "high": 112.0,
                    "low": 110.0,
                    "close": 111.1125,
                    "volume": 49511400,
                    "closeunadj": 444.45,
                },
                {
                    "date": "2020-08-28",
                    "open": 125.0,
                    "high": 126.0,
                    "low": 124.0,
                    "close": 124.8075,
                    "volume": 46907500,
                    "closeunadj": 499.23,
                },
                {
                    "date": "2020-08-31",
                    "open": 127.58,
                    "high": 131.0,
                    "low": 126.0,
                    "close": 129.04,
                    "volume": 225702700,
                    "closeunadj": 129.04,
                },
            ],
            "actions": [
                {"date": "2020-08-07", "action": "dividend", "value": 0.205},
                {"date": "2020-08-31", "action": "split", "value": 4},
            ],
        }
    )

    bars = server.provider().fetch_daily("AAPL", date(2020, 8, 7), date(2020, 8, 31))

    assert server.requests[1].url.path == "/v1.0/data/stocks"
    assert bars["dividend"].to_list() == pytest.approx([0.82, 0.0, 0.0])
    assert bars["close"].to_list() == pytest.approx([444.45, 499.23, 129.04])
    assert bars["split_factor"].to_list() == [1.0, 1.0, 4.0]


def test_pages_are_requested_until_a_short_page() -> None:
    server = uso()

    bars = server.provider(page_size=2).fetch_daily("USO", date(2020, 4, 27), date(2020, 4, 29))

    price_requests = [r for r in server.requests if r.url.path.endswith("/funds")]
    assert [request.url.params["skip"] for request in price_requests] == ["0", "2"]
    assert bars.height == 3


def test_truncated_history_is_rejected() -> None:
    server = uso(tickers=[{"ticker": "USO", "table": "funds", "firstpricedate": "1997-12-31"}])

    with pytest.raises(SharadarError, match="check SHARADAR_API_KEY"):
        server.provider().fetch_daily("USO", date(1990, 1, 1), date(2020, 4, 29))


def test_incremental_windows_are_not_mistaken_for_truncation() -> None:
    server = uso(tickers=[{"ticker": "USO", "table": "funds", "firstpricedate": "1997-12-31"}])

    bars = server.provider().fetch_daily("USO", date(2020, 4, 27), date(2020, 4, 29))

    assert bars.height == 3


def test_unknown_tickers_are_rejected() -> None:
    with pytest.raises(SharadarError, match="no price history"):
        uso(tickers=[]).provider().fetch_daily("USO", date(2020, 4, 27), date(2020, 4, 29))


def test_actions_without_prices_are_rejected() -> None:
    server = uso(actions=[{"date": "2020-04-30", "action": "dividend", "value": 0.1}])

    with pytest.raises(SharadarError, match="corporate actions without prices on 1 rows"):
        server.provider().fetch_daily("USO", date(2020, 4, 27), date(2020, 4, 29))


def test_split_actions_must_match_the_price_adjustment() -> None:
    server = uso(actions=[{"date": "2020-04-28", "action": "split", "value": 2.0}])

    with pytest.raises(SharadarError, match="disagree with split actions"):
        server.provider().fetch_daily("USO", date(2020, 4, 27), date(2020, 4, 29))


def test_no_prices_give_empty_bars() -> None:
    bars = (
        uso(funds=[], actions=[]).provider().fetch_daily("USO", date(2020, 5, 1), date(2020, 5, 1))
    )

    assert bars.is_empty()
    assert bars.schema == BAR_SCHEMA


def test_errors_never_reveal_the_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "Exceeds free tier", "echo": KEY})

    provider = SharadarProvider(KEY, transport=httpx.MockTransport(handler))

    with pytest.raises(SharadarError) as raised:
        provider.fetch_daily("USO", date(2020, 4, 27), date(2020, 4, 29))

    assert "403" in str(raised.value)
    assert "Exceeds free tier" in str(raised.value)
    assert "check SHARADAR_API_KEY" in str(raised.value)
    assert KEY not in str(raised.value)
