import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from quant.live.ledger import LedgerDay

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    date TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    equity REAL NOT NULL,
    peak_equity REAL NOT NULL,
    positions TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    session TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    reference_price REAL NOT NULL,
    status TEXT NOT NULL,
    filled_quantity REAL NOT NULL,
    filled_price REAL,
    commission REAL
);
"""


@dataclass(frozen=True, slots=True)
class OrderRecord:
    client_order_id: str
    session: date
    symbol: str
    quantity: float
    reference_price: float
    status: str
    filled_quantity: float
    filled_price: float | None
    commission: float | None = None


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(SCHEMA)

    def save_day(self, day: LedgerDay) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO ledger VALUES (?, ?, ?, ?, ?)",
                (
                    day.date.isoformat(),
                    day.cash,
                    day.equity,
                    day.peak_equity,
                    json.dumps(day.positions, sort_keys=True),
                ),
            )

    def days(self) -> list[LedgerDay]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM ledger ORDER BY date").fetchall()
        return [
            LedgerDay(date.fromisoformat(day), cash, json.loads(positions), equity, peak)
            for day, cash, equity, peak, positions in rows
        ]

    def latest_day(self) -> LedgerDay | None:
        days = self.days()
        return days[-1] if days else None

    def save_order(self, order: OrderRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    order.client_order_id,
                    order.session.isoformat(),
                    order.symbol,
                    order.quantity,
                    order.reference_price,
                    order.status,
                    order.filled_quantity,
                    order.filled_price,
                    order.commission,
                ),
            )

    def orders(self, session: date) -> list[OrderRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM orders WHERE session = ? ORDER BY symbol", (session.isoformat(),)
            ).fetchall()
        return [
            OrderRecord(
                cid, date.fromisoformat(day), symbol, qty, price, status, filled, fill, commission
            )
            for cid, day, symbol, qty, price, status, filled, fill, commission in rows
        ]

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            yield connection
