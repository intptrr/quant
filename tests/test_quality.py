from datetime import date, timedelta

import polars as pl
import pytest
from factories import make_bars, set_value

from quant.data.quality import Issue, Severity, check_bars, describe_dates

CLOSES = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0]


def clean_bars() -> pl.DataFrame:
    return make_bars(CLOSES)


def findings(issues: list[Issue]) -> set[tuple[str, Severity]]:
    return {(issue.check, issue.severity) for issue in issues}


def test_clean_bars_have_no_issues() -> None:
    bars = clean_bars()

    assert check_bars("SPY", bars, bars["date"][-1]) == []


def test_missing_bars_are_an_error() -> None:
    assert findings(check_bars("SPY", None, None)) == {("missing", Severity.ERROR)}


@pytest.mark.parametrize(
    ("column", "value", "check", "severity"),
    [
        ("close", -1.0, "non_positive_price", Severity.ERROR),
        ("high", 1.0, "high_below_low", Severity.ERROR),
        ("dividend", -0.5, "negative_dividend", Severity.ERROR),
        ("split_factor", 0.0, "non_positive_split", Severity.ERROR),
        ("volume", None, "null_values", Severity.ERROR),
        ("open", 200.0, "open_outside_range", Severity.WARNING),
        ("close", 101.9, "close_outside_range", Severity.WARNING),
        ("volume", 0.0, "zero_volume", Severity.WARNING),
    ],
)
def test_row_checks(column: str, value: float | None, check: str, severity: Severity) -> None:
    bars = set_value(clean_bars(), 2, column, value)

    assert (check, severity) in findings(check_bars("SPY", bars, None))


def test_floating_point_noise_at_the_range_edges_is_ignored() -> None:
    bars = clean_bars()
    noisy = set_value(bars, 2, "close", bars["high"][2] * (1.0 + 1e-15))

    assert check_bars("SPY", noisy, None) == []


def test_extreme_returns_are_flagged() -> None:
    bars = make_bars([100.0, 100.0, 130.0, 130.0])

    assert ("extreme_return", Severity.WARNING) in findings(check_bars("SPY", bars, None))


def test_duplicate_dates_are_an_error() -> None:
    bars = pl.concat([clean_bars(), clean_bars().head(1)]).sort("date")

    assert ("duplicate_date", Severity.ERROR) in findings(check_bars("SPY", bars, None))


def test_missing_sessions_are_a_warning() -> None:
    bars = clean_bars().filter(pl.col("date") != clean_bars()["date"][2])

    assert findings(check_bars("SPY", bars, None)) == {("missing_sessions", Severity.WARNING)}


def test_dates_outside_the_trading_calendar_are_an_error() -> None:
    bars = clean_bars().with_columns(
        pl.when(pl.col("date") == date(2024, 1, 5))
        .then(pl.lit(date(2024, 1, 6)))
        .otherwise(pl.col("date"))
        .alias("date")
    )

    assert ("non_session_dates", Severity.ERROR) in findings(check_bars("SPY", bars, None))


def test_data_ending_before_the_expected_session_is_stale() -> None:
    bars = clean_bars()
    expected_last = bars["date"][-1] + timedelta(days=7)

    assert findings(check_bars("SPY", bars, expected_last)) == {("stale", Severity.WARNING)}


def test_describe_dates_shows_a_sample() -> None:
    days = [date(2024, 1, day) for day in range(2, 7)]

    assert describe_dates(days) == "5 rows (2024-01-02, 2024-01-03, 2024-01-04, ...)"
