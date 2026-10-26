from datetime import date, datetime, time, timedelta
from functools import cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

NEW_YORK = ZoneInfo("America/New_York")
CALENDAR_START = date(1990, 1, 2)
SESSION_GAP_DAYS = 10


@cache
def nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS", start=CALENDAR_START)


def sessions(start: date, end: date) -> list[date]:
    calendar = nyse()
    first = max(start, calendar.first_session.date())
    last = min(end, calendar.last_session.date())
    if first > last:
        return []
    return [session.date() for session in calendar.sessions_in_range(first, last)]


def last_closed_session(now: datetime) -> date:
    calendar = nyse()
    session = calendar.date_to_session(now.astimezone(NEW_YORK).date(), direction="previous")
    if calendar.session_close(session) > now:
        session = calendar.previous_session(session)
    return session.date()


def new_york_today(now: datetime) -> date:
    return now.astimezone(NEW_YORK).date()


def is_session(day: date) -> bool:
    return sessions(day, day) == [day]


def previous_session(day: date) -> date:
    return sessions(day - timedelta(days=SESSION_GAP_DAYS), day - timedelta(days=1))[-1]


def next_session(day: date) -> date:
    return sessions(day + timedelta(days=1), day + timedelta(days=SESSION_GAP_DAYS))[0]


def at_new_york_time(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, NEW_YORK)
