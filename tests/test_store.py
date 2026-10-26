from pathlib import Path

from factories import make_bars

from quant.data.store import BarStore


def test_written_bars_read_back_unchanged(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    bars = make_bars([100.0, 101.0, 102.0])

    store.write("sharadar", "SPY", bars)

    stored = store.read("sharadar", "SPY")
    assert stored is not None
    assert stored.equals(bars)


def test_reading_an_unknown_symbol_returns_none(tmp_path: Path) -> None:
    assert BarStore(tmp_path).read("sharadar", "SPY") is None


def test_write_leaves_only_the_final_file(tmp_path: Path) -> None:
    store = BarStore(tmp_path)

    store.write("sharadar", "SPY", make_bars([100.0]))

    assert [path.name for path in store.path("sharadar", "SPY").parent.iterdir()] == ["SPY.parquet"]
