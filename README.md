# quant

A personal system for US ETF portfolios: daily data, a backtest, and paper trading at the opening auction through
IBKR or Alpaca. A strategy turns prices into target weights; paper trading runs a 60/40 SPY/IEF portfolio.

## Setup

```bash
uv sync
cp .env.example .env   # add SHARADAR_API_KEY, plus the Alpaca keys for paper trading
```

## Commands

```bash
uv run quant data sync                 # daily bars for every strategy's symbols and SPY, the Fed funds rate, quality checks; --full re-downloads
uv run quant data check                # quality checks only
uv run quant backtest                  # compare every strategy with SPY buy-and-hold, in sample only; --output DIR saves the details
uv run quant backtest 'sixty-*' spy    # only the strategies with these names or patterns
uv run quant backtest spy --final      # also show the out-of-sample period, once a strategy is chosen
uv run quant paper check               # broker connection, account and market data
uv run quant paper pre-open --dry-run  # orders for the next opening auction, nothing submitted
```

`data` and `backtest` take strategy names or patterns as arguments and default to all strategies. `paper` commands
trade the strategy named in `configs/trading.toml` (`strategy = "sixty-forty"`) unless `--strategy` names another.
Every command reads `configs/trading.toml` unless `--trading` points elsewhere.

## Strategies

A strategy has `symbols` and a `targets(prices)` method that returns a target `weight` and a rebalance `band` for each
symbol and day (`Strategy` in `src/quant/strategies/__init__.py`). Each type of strategy is a module in
`src/quant/strategies/`, and its parameters live in the TOML file of the same name in `configs/strategies/`, one
table per named variant. Names are unique across files, and a parameter that is missing, unknown or of the wrong type
fails when the files load.

The only type so far is `fixed` (`FixedWeights`), in `configs/strategies/fixed.toml`:

```toml
[spy]
weights = { SPY = 1.0 }

[sixty-forty]
weights = { SPY = 0.6, IEF = 0.4 }
```

A new variant is a new table in that file. A new type is a module in `src/quant/strategies/`, a line in `TYPES` in
`src/quant/strategies/__init__.py` and a TOML file named after it. Every variant is tested on synthetic prices to
check that its targets never depend on later prices.

Weights are fractions of equity: negative weights are shorts, and weights adding up to more than 1 borrow cash. The
portfolio starts trading once every symbol has a price. A position trades only when its weight leaves its band,
`rebalance_band` (default 0) times its target weight on either side, and then only back to the nearest edge. With a
band of 0.1, a 100% SPY target holds anything from 90% to 110%, and starting from cash it buys 90%. With a band of 0,
positions go back to their targets whenever the order is at least `min_order_value` (`configs/trading.toml`).

## Backtest

Targets are set at the close of each day and filled at the next day's open, on prices adjusted for splits and
dividends. The costs come from `configs/trading.toml`:

- slippage, a flat `slippage_bps` on traded value;
- commissions and fees from the broker's schedule, for an account of `capital` dollars;
- a borrow fee on shorts;
- interest: borrowed cash pays the Fed funds rate plus the broker's margin spread, and idle cash above the broker's
  threshold earns the Fed funds rate minus its spread.

The report compares the strategies with SPY buy-and-hold (dividends reinvested, no costs) over the period they all
cover: CAGR, volatility, Sharpe ratio (of returns above the Fed funds rate) and maximum drawdown, each strategy's
average leverage, turnover and costs, and yearly returns.

Research keeps the years from 2016-01-04 on (`OUT_OF_SAMPLE_START` in `src/quant/cli.py`) as an out-of-sample period.
`quant backtest` only runs on the prices before that date, so strategies are developed and compared without seeing
those years. `--final` adds the full and out-of-sample periods, for one check of a chosen strategy. New variants are
committed before they are run, variants that have run are not edited (a change gets a new name), and rejected ones
stay in the files, so the history shows what was tried and when.

## Paper trading

Orders go to the opening auction through IBKR (the default) or Alpaca, chosen in `configs/trading.toml` or with
`--broker`. Broker setup, the daily routine and the safety checks are in [`docs/operations.md`](docs/operations.md).

## Data

- Sharadar (`api.sharadar.com`): prices from the `funds` (or `stocks`) table and dividends and splits from the
  `actions` table, stored unadjusted in `data/raw/sharadar/<SYMBOL>.parquet`. Adjusted prices are computed locally.
  History starts on 1997-12-31. Each sync compares against the first price date in the `tickers` table and fails if
  the history is truncated (an invalid key or a subscription that covers too few years).
- FRED: the effective Fed funds rate (series DFF, no key needed) in `data/raw/fred/DFF.parquet`. It sets interest on
  borrowed and idle cash in the backtest and the paper ledger.
- Paper trading state (the shadow ledger and orders) lives in `data/state/paper-<broker>.sqlite3`.

## Layout

| Path | Contents |
|---|---|
| `configs/strategies/` | Strategy parameters: one file per type, one table per named variant |
| `configs/trading.toml` | Strategy traded, broker, account size, costs, minimum order and risk limits |
| `src/quant/strategies` | Strategy interface, the strategy types and the loading of their parameters |
| `src/quant/data` | Sharadar and FRED downloads, Parquet storage, adjustment, quality checks, incremental sync |
| `src/quant/backtest` | Daily backtest: targets at the close, fills at the next open, costs by type |
| `src/quant/analytics` | Performance metrics and the backtest report |
| `src/quant/core` | Trading calendar, trading config and the rebalancing rule |
| `src/quant/execution` | Broker interfaces (IBKR, Alpaca) |
| `src/quant/live` | Paper trading: shadow ledger, order planning, risk checks, state storage, daily jobs |

## Development

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
uv run pyright
```
