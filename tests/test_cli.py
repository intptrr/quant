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
from quant.live.jobs import Desk, JobReport

PROJECT = Path(__file__).parents[1]
TRADING = PROJECT / "configs" / "trading.toml"
runner = CliRunner()


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BarStore:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    RateStore(tmp_path / "data").write(flat_rates(0.04))
    return BarStore(tmp_path / "data")


def write_strategies(**weights: str) -> None:
    folder = Path("configs/strategies")
    folder.mkdir(parents=True)
    sections = [f"[{name}]\nweights = {{ {weight} }}\n" for name, weight in weights.items()]
    (folder / "fixed.toml").write_text("\n".join(sections))


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


def test_check_covers_the_strategies_and_the_benchmark(store: BarStore) -> None:
    write_strategies(qqq="QQQ = 1.0", ief="IEF = 1.0")

    result = runner.invoke(app, ["data", "check"])

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
    write_strategies(balanced="SPY = 0.6, IEF = 0.4", stocks="SPY = 1.0", bonds="IEF = 1.0")

    result = runner.invoke(app, backtest("balanced", "stocks"))

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
    write_strategies(spy="SPY = 1.0")

    result = runner.invoke(app, backtest("--final"))

    assert result.exit_code == 0, result.output
    titles = [section.splitlines()[0] for section in result.output.split("\n\n")]
    assert titles[0].startswith("Full period: 2015-01-05 to 2016-")
    assert titles[1] == "In sample: 2015-01-05 to 2015-12-31"
    assert titles[2].startswith("Out of sample: 2016-01-04 to 2016-")
    assert "hidden" not in result.output


def test_backtest_writes_each_strategys_results(store: BarStore) -> None:
    write_history(store, "SPY", "IEF")
    write_strategies(spy="SPY = 1.0", ief="IEF = 1.0")

    result = runner.invoke(app, backtest("--output", "results"))

    assert result.exit_code == 0, result.output
    assert sorted(
        path.relative_to("results").as_posix() for path in Path("results").rglob("*.*")
    ) == [
        "ief/daily.parquet",
        "ief/positions.parquet",
        "spy/daily.parquet",
        "spy/positions.parquet",
    ]


def test_strategies_without_in_sample_prices_are_rejected(store: BarStore) -> None:
    write_history(store, "SPY", start=date(2016, 2, 1))
    write_strategies(spy="SPY = 1.0")

    result = runner.invoke(app, backtest())

    assert result.exit_code == 2
    assert "spy has no prices before 2016-01-04" in plain(result.output)


def test_backtest_needs_strategies(store: BarStore) -> None:
    result = runner.invoke(app, backtest())

    assert result.exit_code == 2
    assert "no strategies in configs/strategies" in plain(result.output)


def test_paper_trades_the_strategy_named_in_the_trading_config(
    store: BarStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_strategies(spy="SPY = 1.0", ief="IEF = 1.0")
    traded: list[str] = []

    def check(desk: Desk) -> JobReport:
        traded.extend(desk.strategy.symbols)
        return JobReport("check", [])

    monkeypatch.setattr("quant.cli._alpaca_keys", lambda settings: ("key", "secret"))
    monkeypatch.setattr("quant.cli.sharadar_provider", lambda settings: None)
    monkeypatch.setattr("quant.cli._broker", lambda settings, name: FakeBroker())
    monkeypatch.setattr("quant.cli.broker_check", check)

    runner.invoke(app, ["paper", "check", "--trading", str(TRADING)])
    runner.invoke(app, ["paper", "check", "--strategy", "ief", "--trading", str(TRADING)])

    assert traded == ["SPY", "IEF"]


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
