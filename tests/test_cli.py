from datetime import date
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


def write_portfolio(weights: str, name: str = "portfolio") -> Path:
    path = Path(f"{name}.toml")
    path.write_text(f"[weights]\n{weights}\n")
    return path


def write_history(store: BarStore, *symbols: str, start: date = date(2015, 1, 2)) -> None:
    for seed, symbol in enumerate(symbols):
        bars = make_bars(random_walk(seed, 400, drift=0.0003), start=start)
        store.write("sharadar", symbol, bars)


def backtest(*arguments: str) -> list[str]:
    return ["backtest", *arguments, "--trading", str(TRADING)]


def plain(output: str) -> str:
    return " ".join(output.replace("│", " ").split())


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


def test_backtest_compares_strategies_with_spy_in_sample_only(store: BarStore) -> None:
    write_history(store, "SPY", "IEF")
    balanced = write_portfolio("SPY = 0.6\nIEF = 0.4", "balanced")
    stocks = write_portfolio("SPY = 1.0", "stocks")

    result = runner.invoke(app, backtest(str(balanced), str(stocks)))

    assert result.exit_code == 0, result.output
    title, header, *rows = result.output.split("\n\n")[0].splitlines()
    assert title == "In sample: 2015-01-05 to 2015-12-31"
    assert header.split() == ["Portfolio", "CAGR", "Vol", "Sharpe", "MaxDD"]
    assert [row.split()[0] for row in rows] == ["balanced", "stocks", "SPY"]
    assert "correlation to SPY" in result.output
    assert "Out of sample" not in result.output
    assert result.output.rstrip().endswith(
        "Results from 2016-01-04 on are hidden; add --final to see them."
    )


def test_backtest_final_adds_the_full_and_out_of_sample_periods(store: BarStore) -> None:
    write_history(store, "SPY")

    result = runner.invoke(app, backtest(str(PORTFOLIO), "--final"))

    assert result.exit_code == 0, result.output
    titles = [section.splitlines()[0] for section in result.output.split("\n\n")]
    assert titles[0].startswith("Full period: 2015-01-05 to 2016-")
    assert titles[1] == "In sample: 2015-01-05 to 2015-12-31"
    assert titles[2].startswith("Out of sample: 2016-01-04 to 2016-")
    assert "hidden" not in result.output


def test_backtest_writes_each_strategys_results(store: BarStore) -> None:
    write_history(store, "SPY")

    result = runner.invoke(app, backtest(str(PORTFOLIO), "--output", "results"))

    assert result.exit_code == 0, result.output
    assert sorted(
        path.relative_to("results").as_posix() for path in Path("results").rglob("*.*")
    ) == [
        "portfolio/daily.parquet",
        "portfolio/positions.parquet",
    ]


def test_strategies_without_in_sample_prices_are_rejected(store: BarStore) -> None:
    write_history(store, "SPY", start=date(2016, 2, 1))

    result = runner.invoke(app, backtest(str(write_portfolio("SPY = 1.0"))))

    assert result.exit_code == 2
    assert "portfolio.toml has no prices before 2016-01-04" in plain(result.output)


def test_strategy_file_names_must_differ(store: BarStore) -> None:
    for folder in ("a", "b"):
        Path(folder).mkdir()
        Path(folder, "portfolio.toml").write_text("[weights]\nSPY = 1.0\n")

    result = runner.invoke(app, backtest("a/portfolio.toml", "b/portfolio.toml"))

    assert result.exit_code == 2
    assert "strategy file names must differ" in plain(result.output)


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
