from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import polars as pl

from quant.core.calendar import sessions
from quant.data.adjust import adjust
from quant.data.schema import BAR_SCHEMA, PRICE_COLUMNS

MAX_ABS_DAILY_RETURN = 0.25
RANGE_TOLERANCE = 1e-9
SAMPLE_SIZE = 3


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Issue:
    symbol: str
    severity: Severity
    check: str
    detail: str


def outside_range(column: str) -> pl.Expr:
    return (pl.col(column) > pl.col("high") * (1.0 + RANGE_TOLERANCE)) | (
        pl.col(column) < pl.col("low") * (1.0 - RANGE_TOLERANCE)
    )


ROW_CHECKS: tuple[tuple[str, Severity, pl.Expr], ...] = (
    ("null_values", Severity.ERROR, pl.any_horizontal(pl.all().is_null())),
    ("duplicate_date", Severity.ERROR, pl.col("date").is_duplicated()),
    ("non_positive_price", Severity.ERROR, pl.min_horizontal(*PRICE_COLUMNS) <= 0),
    ("high_below_low", Severity.ERROR, pl.col("high") < pl.col("low")),
    ("negative_dividend", Severity.ERROR, pl.col("dividend") < 0),
    ("non_positive_split", Severity.ERROR, pl.col("split_factor") <= 0),
    ("open_outside_range", Severity.WARNING, outside_range("open")),
    ("close_outside_range", Severity.WARNING, outside_range("close")),
    ("zero_volume", Severity.WARNING, pl.col("volume") <= 0),
)


def check_bars(symbol: str, bars: pl.DataFrame | None, expected_last: date | None) -> list[Issue]:
    if bars is None or bars.is_empty():
        return [Issue(symbol, Severity.ERROR, "missing", "no bars stored")]
    if bars.schema != BAR_SCHEMA:
        return [Issue(symbol, Severity.ERROR, "schema", f"unexpected schema {bars.schema}")]
    return [
        *_row_issues(symbol, bars),
        *_return_issues(symbol, bars),
        *_calendar_issues(symbol, bars),
        *_staleness_issues(symbol, bars, expected_last),
    ]


def describe_dates(dates: Sequence[date]) -> str:
    sample = ", ".join(day.isoformat() for day in dates[:SAMPLE_SIZE])
    suffix = ", ..." if len(dates) > SAMPLE_SIZE else ""
    return f"{len(dates)} rows ({sample}{suffix})"


def _row_issues(symbol: str, bars: pl.DataFrame) -> list[Issue]:
    issues = []
    for check, severity, predicate in ROW_CHECKS:
        dates = bars.filter(predicate)["date"].to_list()
        if dates:
            issues.append(Issue(symbol, severity, check, describe_dates(dates)))
    return issues


def _return_issues(symbol: str, bars: pl.DataFrame) -> list[Issue]:
    extreme = adjust(bars).filter(pl.col("total_return").abs() > MAX_ABS_DAILY_RETURN)
    if extreme.is_empty():
        return []
    return [
        Issue(symbol, Severity.WARNING, "extreme_return", describe_dates(extreme["date"].to_list()))
    ]


def _calendar_issues(symbol: str, bars: pl.DataFrame) -> list[Issue]:
    actual = set(bars["date"].to_list())
    expected = set(sessions(min(actual), max(actual)))
    issues = []
    if missing := sorted(expected - actual):
        issues.append(Issue(symbol, Severity.WARNING, "missing_sessions", describe_dates(missing)))
    if extra := sorted(actual - expected):
        issues.append(Issue(symbol, Severity.ERROR, "non_session_dates", describe_dates(extra)))
    return issues


def _staleness_issues(symbol: str, bars: pl.DataFrame, expected_last: date | None) -> list[Issue]:
    last = bars["date"].max()
    if expected_last is None or not isinstance(last, date) or last >= expected_last:
        return []
    detail = f"last bar {last.isoformat()}, expected {expected_last.isoformat()}"
    return [Issue(symbol, Severity.WARNING, "stale", detail)]
