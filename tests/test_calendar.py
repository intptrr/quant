from datetime import date, datetime

from quant.core.calendar import (
    NEW_YORK,
    is_session,
    last_closed_session,
    next_session,
    previous_session,
    sessions,
)


def test_sessions_skip_weekends_and_holidays() -> None:
    assert sessions(date(2024, 7, 3), date(2024, 7, 8)) == [
        date(2024, 7, 3),
        date(2024, 7, 5),
        date(2024, 7, 8),
    ]


def test_sessions_are_empty_for_a_weekend() -> None:
    assert sessions(date(2024, 7, 6), date(2024, 7, 7)) == []


def test_last_closed_session_before_the_close_is_the_previous_session() -> None:
    assert last_closed_session(datetime(2024, 7, 5, 15, 0, tzinfo=NEW_YORK)) == date(2024, 7, 3)


def test_last_closed_session_after_the_close_is_today() -> None:
    assert last_closed_session(datetime(2024, 7, 5, 16, 30, tzinfo=NEW_YORK)) == date(2024, 7, 5)


def test_last_closed_session_on_a_weekend_is_friday() -> None:
    assert last_closed_session(datetime(2024, 7, 7, 12, 0, tzinfo=NEW_YORK)) == date(2024, 7, 5)


def test_last_closed_session_respects_early_closes() -> None:
    assert last_closed_session(datetime(2024, 11, 29, 13, 30, tzinfo=NEW_YORK)) == date(
        2024, 11, 29
    )


def test_is_session() -> None:
    assert is_session(date(2024, 7, 3))
    assert not is_session(date(2024, 7, 4))
    assert not is_session(date(2024, 7, 6))


def test_previous_session_skips_holidays_and_weekends() -> None:
    assert previous_session(date(2024, 7, 5)) == date(2024, 7, 3)
    assert previous_session(date(2024, 7, 8)) == date(2024, 7, 5)
    assert previous_session(date(2024, 7, 7)) == date(2024, 7, 5)


def test_next_session_skips_holidays_and_weekends() -> None:
    assert next_session(date(2024, 7, 3)) == date(2024, 7, 5)
    assert next_session(date(2024, 7, 5)) == date(2024, 7, 8)
    assert next_session(date(2024, 7, 6)) == date(2024, 7, 8)
