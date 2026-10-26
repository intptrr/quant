# Paper trading runbook

## Brokers

The portfolio in `configs/portfolio.toml` trades at the opening auction through one broker at a time:

- **IBKR** (the default, `broker = "ibkr"` in `configs/trading.toml`): the intended live venue, IBKR Pro Tiered pricing.
- **Alpaca**: a commission-free paper account that needs no local software, handy for quick tests.

Any `paper` command takes `--broker alpaca` or `--broker ibkr` to override the configured broker. Each broker keeps
its own shadow ledger in `data/state/paper-<broker>.sqlite3`, so switching never mixes the two.

## Prerequisites

- `.env` contains:
  - `SHARADAR_API_KEY` for prices;
  - `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. These are needed with either broker: announced splits and the latest
    prices for the safety checks come from Alpaca's free market data;
  - for IBKR, `IBKR_ACCOUNT`: the paper account ID, which starts with `DU`. `IBKR_HOST`, `IBKR_PORT` and
    `IBKR_CLIENT_ID` default to `127.0.0.1`, `4002` (IB Gateway's paper port) and `17`.
- `uv run quant data sync` has run at least once. Besides prices, it downloads the Fed funds rate from FRED, which
  sets interest on borrowed and idle cash. `pre-open` doesn't refresh rates, so rerun `data sync` now and then; weekly
  is plenty.
- Reports print to the terminal only. When something needs a human, the report title starts with `[ATTENTION]` and
  the command exits with a non-zero status.

## One-time IBKR setup

1. In Client Portal, open the paper trading account settings:
   - set its balance to $100,000 if IBKR lets you choose; otherwise start the ledger with
     `uv run quant paper init --capital 100000`;
   - share your live account's market data with it. IBKR's simulator needs quotes to fill orders, and the short
     availability check reads market data;
   - note the paper username and account ID.
2. IBKR paper accounts follow the live account's pricing plan, so set the live account to IBKR Pro Tiered.
3. Install IB Gateway and log in with the paper username. In its API settings, use socket port 4002, turn off
   "Read-Only API" and allow connections from localhost only.
4. Add `IBKR_ACCOUNT` to `.env`.
5. Run `uv run quant paper check`. It connects to the broker and shows the account equity, positions, open orders, the
   symbols that can't be shorted, and whether market data arrives. Any problem shows up as `[ATTENTION]`.

IB Gateway (or TWS) must be running and logged in whenever a `paper` command talks to IBKR.

## Dry runs (no orders)

```bash
uv run quant paper pre-open --dry-run
```

Syncs data, computes target positions, runs the risk checks and lists the orders for the next opening auction. A dry
run submits nothing, never writes the shadow ledger and doesn't need `init`: before initialization it plans from a
flat account with the broker's account equity.

## Going live

1. Make sure the paper account has no positions and no open orders.
2. Run `uv run quant paper init` when you are ready to place the first orders. The shadow ledger starts at the last
   closed session, with the account equity as starting capital (or `--capital`). The paper-vs-backtest comparison
   starts on that day, so don't initialize days in advance.
3. Run `uv run quant paper pre-open` inside the submission window.

## Daily routine

Opening-auction orders are accepted only from 19:00 ET on the previous session until 09:28 ET. By default `pre-open`
prepares the next opening auction that hasn't happened yet, so an evening run prepares the next session.

| When (Pacific / Eastern) | Command | What it does |
| --- | --- | --- |
| From 16:00 PT on a session day until 06:28 PT on the next session (19:00–09:28 ET) | `uv run quant paper pre-open` | Syncs and checks data; settles the shadow ledger through the previous session (fills, commissions, dividends, splits, borrow fees and interest) and reports fill slippage; reconciles with the broker; computes target positions; runs the risk checks; submits opening-auction orders |
| After the open, before that evening's `pre-open` | `uv run quant paper post-open` | Records the opening fills and reconciles early. Optional at Alpaca; see the IBKR note below |
| Any time | `uv run quant paper report` | Returns and tracking error of the paper account against a backtest replay (same start date, also starting flat), plus slippage against the open for each symbol |

- **IBKR keeps fills only for the current day.** Its API returns executions only until midnight ET (21:00 PT), so the
  day's fills must be recorded the same day. Either run `pre-open` before 21:00 PT, or run `post-open` once after the
  open (06:30–21:00 PT) and then `pre-open` whenever you like. If neither happens, `pre-open` stops and says so.
- At Alpaca, `pre-open` alone is enough: it settles today's fills before preparing the next session.
- Sharadar updates its data twice a day, at 17:30 and 23:30 ET; runs after 20:30 PT pick up the second update.
- Weekends and holidays are fine: the default finds the next session. Rerunning for the same session never submits
  duplicate orders.
- A missed day: the next `pre-open` settles the missed sessions in order, but the missed day itself isn't rebalanced.

## Orders and costs

- Orders are whole shares, sized from the last close and rounded toward zero, so a position never goes past its
  target.
- Orders under $1,000 (`min_order_value` in `configs/trading.toml`) are skipped unless they close a position, because
  IBKR's per-order minimum commission would make them expensive. The report lists the symbols that were skipped.
- The ledger books the commission the broker reports for each fill. When the broker reports none, as Alpaca doesn't,
  it uses the commission schedule of the broker in `configs/trading.toml`.
- Borrowed cash pays the Fed funds rate plus the broker's margin spread. At IBKR, idle cash above $10,000 earns the
  Fed funds rate − 0.5%.

## Safety checks

`pre-open` stops before placing any order and marks the report `[ATTENTION]` when:

- it runs outside the submission window;
- the data isn't current or a quality check reports errors;
- the shadow ledger's positions differ from the broker's (for example after a split or a manual trade).

Individual orders are rejected, also marked `[ATTENTION]`, when:

- a split or stock dividend takes effect at the next open, according to Alpaca's corporate actions data. The order
  would be sized from pre-split prices, so it waits one session, until prices and the ledger reflect the split.
  Positions held through a split are adjusted in the ledger from Sharadar's split factor and checked against the
  broker at the next reconciliation. IBKR's paper account doesn't process splits, so a split there shows up as a
  position mismatch.
- the order's value at the last close exceeds 100% of equity (`max_order_fraction`). Shares are rounded toward zero,
  so a 100% target never needs more than the account, and this limit only catches order errors. Orders fill at the
  open, so a gap up can still leave cash slightly negative for a day.
- the latest price is more than 10% away from the last close, which usually means a split or bad data.
- drawdown exceeds 25% (`max_drawdown`); after that only orders that reduce positions go through.
- the broker refuses the order, for example for insufficient buying power. The other orders still go out; rerun
  `pre-open` to retry it once the cause is fixed.

Shorts in symbols the broker can't easily lend are skipped. That's listed in the report without `[ATTENTION]`, but
the backtest replay still holds those shorts, so paper results will differ from it.

## When you see `[ATTENTION]`

1. Read the reason in the report.
2. Position mismatch: find the cause in the broker's account view (split, partial fill, manual trade), fix it, then
   rerun `pre-open`.
3. Data problems: investigate with `uv run quant data check`.
4. Order refused by the broker: fix the cause given in the report (for example buying power), then rerun `pre-open`.
5. Don't place manual orders until it's resolved, or the ledger and the broker drift further apart.

## Passing the paper run

The paper run passes when the IBKR paper account meets all of these, checked with the daily `pre-open` output and
`uv run quant paper report`:

1. 20 trading sessions in a row with `pre-open` run inside the submission window. A session without a run restarts
   the count.
2. The ledger matches the broker's positions at every `pre-open`, or any mismatch has a known cause such as a split.
3. Every `[ATTENTION]` has a known cause and was handled as described above.
4. Tracking error against the backtest replay is under 1% a year.

The slippage table is for information only: paper fills are simulated, so they say little about real trading costs.
