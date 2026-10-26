from pathlib import Path
from types import SimpleNamespace

import pytest
from factories import flat_rates, make_bars, random_walk, set_value
from fakes import FakeBroker, FakeSharadar
from typer.testing import CliRunner

from quant.cli import app
from quant.data.rates import RateStore
from quant.data.store import BarStore
from quant.live.jobs import JobReport

PROJECT = Path(__file__).parents[1]
PORTFOLIO = PROJECT / "configs" / "portfolio.toml"
TRADING = PROJECT / "configs" / "trading.toml"
runner = CliRunner()


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BarStore:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    RateStore(tmp_path / "data").write(flat_rates(0.04))
    return BarStore(tmp_path / "data")


def write_portfolio(weights: str) -> Path:
    path = Path("portfolio.toml")
    path.write_text(f"[weights]\n{weights}\n")
    return path


def write_history(store: BarStore, *symbols: str) -> None:
    for seed, symbol in enumerate(symbols):
        store.write("sharadar", symbol, make_bars(random_walk(seed, 400, drift=0.0003)))


def backtest(*options: str) -> list[str]:
    return ["backtest", "--strategy", str(PORTFOLIO), "--trading", str(TRADING), *options]


def test_check_passes_with_only_warnings(store: BarStore) -> None:
    store.write("sharadar", "SPY", make_bars([100.0, 101.0, 102.0]))

    result = runner.invoke(app, ["data", "check", "--symbol", "spy"])

    assert result.exit_code == 0, result.output
    assert "warning SPY    stale" in result.output
    assert "0 errors, 1 warnings" in result.output


def test_check_fails_on_errors(store: BarStore) -> None:
    store.write("sharadar", "SPY", set_value(make_bars([100.0, 101.0, 102.0]), 1, "close", -1.0))

    result = runner.invoke(app, ["data", "check", "--symbol", "SPY"])

    assert result.exit_code == 1
    assert "error   SPY    non_positive_price" in result.output


def test_check_covers_the_portfolio_and_the_benchmark(store: BarStore) -> None:
    portfolio = write_portfolio("QQQ = 0.6\nIEF = 0.4")

    result = runner.invoke(app, ["data", "check", "--strategy", str(portfolio)])

    assert result.exit_code == 1
    missing = [line.split()[1] for line in result.output.splitlines() if " missing " in line]
    assert missing == ["QQQ", "IEF", "SPY"]


def test_sync_requires_an_api_key(store: BarStore) -> None:
    result = runner.invoke(app, ["data", "sync", "--symbol", "SPY"])

    assert isinstance(result.exception, ValueError)
    assert "SHARADAR_API_KEY is not set" in str(result.exception)


def test_sync_downloads_bars_and_rates_then_checks_them(
    store: BarStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    bar = {"open": 100.0, "high": 101.0, "low": 99.0, "volume": 1000}
    server = FakeSharadar(
        {
            "tickers": [{"ticker": "SPY", "table": "funds", "firstpricedate": "2024-01-02"}],
            "funds": [
                {"date": "2024-01-02", "close": 100.0, "closeunadj": 100.0, **bar},
                {"date": "2024-01-03", "close": 100.5, "closeunadj": 100.5, **bar},
            ],
            "actions": [],
        }
    )
    monkeypatch.setattr("quant.cli.sharadar_provider", lambda settings: server.provider())
    monkeypatch.setattr("quant.cli.fetch_benchmark_rates", lambda: flat_rates(0.05))

    result = runner.invoke(app, ["data", "sync", "--symbol", "SPY"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0].split() == [
        "SPY",
        "rows=2",
        "added=2",
        "revised=0",
        "written=True",
    ]
    assert "Fed funds rows=1 last=1990-01-02" in result.output
    assert "warning SPY    stale" in result.output
    bars = store.read("sharadar", "SPY")
    assert bars is not None and bars.height == 2


def test_backtest_compares_the_strategy_with_spy(store: BarStore) -> None:
    write_history(store, "SPY", "IEF")
    portfolio = write_portfolio("SPY = 0.6\nIEF = 0.4")

    result = runner.invoke(
        app,
        [
            "backtest",
            "--strategy",
            str(portfolio),
            "--trading",
            str(TRADING),
            "--output",
            "results",
        ],
    )

    assert result.exit_code == 0, result.output
    title, header, strategy, benchmark, *_ = result.output.splitlines()
    assert title.startswith("Full period: 2024-01-03 to ")
    assert header.split() == ["Portfolio", "CAGR", "Vol", "Sharpe", "MaxDD"]
    assert strategy.startswith("portfolio ")
    assert benchmark.startswith("SPY buy & hold ")
    assert "correlation to SPY" in result.output
    assert sorted(path.name for path in Path("results").iterdir()) == [
        "daily.parquet",
        "positions.parquet",
    ]


def test_backtest_can_split_in_and_out_of_sample(store: BarStore) -> None:
    write_history(store, "SPY")

    result = runner.invoke(app, backtest("--split", "2025-01-02"))

    assert result.exit_code == 0, result.output
    assert "In sample: 2024-01-03 to 2024-12-31" in result.output
    assert "Out of sample: 2025-01-02 to " in result.output


def test_the_split_must_fall_inside_the_backtest(store: BarStore) -> None:
    write_history(store, "SPY")

    result = runner.invoke(app, backtest("--split", "2030-01-02"))

    assert result.exit_code == 2
    assert "must fall inside the backtest period" in result.output


def test_paper_reports_needing_attention_fail_the_command(
    store: BarStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("quant.cli._desk", lambda *args: SimpleNamespace(broker=FakeBroker()))
    monkeypatch.setattr(
        "quant.cli.tracking_report", lambda desk: JobReport("paper tracking", ["lagging"], ok=False)
    )

    result = runner.invoke(app, ["paper", "report", "--trading", str(TRADING)])

    assert result.exit_code == 1
    assert result.output == "[ATTENTION] paper tracking\nlagging\n"


def test_paper_commands_require_alpaca_keys(
    store: BarStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    result = runner.invoke(
        app, ["paper", "post-open", "--session", "2026-10-26", "--trading", str(TRADING)]
    )

    assert isinstance(result.exception, ValueError)
    assert "ALPACA_API_KEY and ALPACA_SECRET_KEY are not set" in str(result.exception)
