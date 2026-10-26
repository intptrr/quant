# quant

A personal system for US ETF portfolios: daily data, a backtest, and paper trading at the opening auction through
IBKR or Alpaca. A strategy turns prices into target weights; the default is a placeholder that holds 100% SPY.

## Setup

```bash
uv sync
cp .env.example .env   # add SHARADAR_API_KEY, plus the Alpaca keys for paper trading
```

## Commands

```bash
uv run quant data sync                    # daily bars for the portfolio and SPY, the Fed funds rate, quality checks; --full re-downloads
uv run quant data check                   # quality checks only
uv run quant backtest                     # backtest the portfolio against SPY buy-and-hold; --output DIR saves the details
uv run quant backtest --split 2016-01-04  # also report the periods before (in sample) and from (out of sample) a date
uv run quant paper check                  # broker connection, account and market data
uv run quant paper pre-open --dry-run     # orders for the next opening auction, nothing submitted
```

Commands read `configs/portfolio.toml` and `configs/trading.toml` unless `--strategy` or `--trading` points elsewhere.

## Strategy

A strategy has `symbols` and a `targets(prices)` method that returns a target `weight` and a rebalance `band` for each
symbol and day (`src/quant/strategy.py`). The only strategy is `FixedWeights`, configured in `configs/portfolio.toml`:

```toml
rebalance_band = 0.0

[weights]
SPY = 1.0
```

Weights are fractions of equity: negative weights are shorts, and weights adding up to more than 1 borrow cash. The
portfolio starts trading once every symbol has a price. A position trades only when its weight leaves its band,
`rebalance_band` times its target weight on either side, and then only back to the nearest edge. With a band of 0.1, a
100% SPY target holds anything from 90% to 110%, and starting from cash it buys 90%. With a band of 0, positions go
back to their targets whenever the order is at least `min_order_value` (`configs/trading.toml`).

## Backtest

Targets are set at the close of each day and filled at the next day's open, on prices adjusted for splits and
dividends. The costs come from `configs/trading.toml`:

- slippage, a flat `slippage_bps` on traded value;
- commissions and fees from the broker's schedule, for an account of `capital` dollars;
- a borrow fee on shorts;
- interest: borrowed cash pays the Fed funds rate plus the broker's margin spread, and idle cash above the broker's
  threshold earns the Fed funds rate minus its spread.

The report compares the portfolio with SPY buy-and-hold (dividends reinvested, no costs) over their common period:
CAGR, volatility, Sharpe ratio (of returns above the Fed funds rate) and maximum drawdown, the portfolio's average
leverage, turnover and costs, and yearly returns.

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
| `configs/portfolio.toml` | Target weights and rebalance band |
| `configs/trading.toml` | Broker, account size, costs, minimum order and risk limits |
| `src/quant/strategy.py` | Strategy interface and the fixed-weight strategy |
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
