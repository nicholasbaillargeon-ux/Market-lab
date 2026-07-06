# marketlab

A market-data pipeline and backtester built around a single thesis:

> **Naive backtests lie.** The strategy is the easy part; the hard part is not
> fooling yourself. This project is organized around the specific ways a
> backtest overstates reality — and the code and tests that neutralize each one.

It ingests real daily bars (Polygon.io, with a keyless yfinance fallback),
stores them in a partitioned **Parquet** lake plus a **Postgres** system of
record, and runs strategies through **two independent engines** — a fast
vectorized one and an accurate event-driven one — that agree to machine
precision, so a fast research result can always be trusted against a realistic
simulation.

```
$ make demo        # runs on real SPY data, no API key required

== 1. Engine parity (zero cost) ==
vectorized              +198.7%   +8.1%    0.64   -33.7%     13.0     13
event_driven            +198.7%   +8.1%    0.64   -33.7%     13.0     13
   -> terminal-return gap: 2.22e-15  (verification: engines agree)

== 2. Look-ahead bias (same strategy, dishonest fill) ==
honest (next open)      +198.7%   +8.1%    0.64   -33.7%     13.0     13
look-ahead (same close) +217.9%   +8.6%    0.67   -33.7%     13.0     13
   -> look-ahead inflates total return by +19.2% of pure fantasy
```

---

## The pitfalls (the whole point)

Each of these is a way a backtest tells you a strategy works when it doesn't.
For each, this repo has a concrete mechanism and a test that fails if the
mechanism regresses.

### 1. Look-ahead bias — *using information you couldn't have had*

The most common and most fatal bug. You compute a signal from today's close and
then "trade" at that same close — but at 4:00pm when you know the close, the
close has already happened. Every engine here enforces **decide at close *t*,
fill at open *t+1*** as a structural rule, not a convention you have to
remember.

- **Mechanism:** `engine/vectorized.py` shifts target weights one bar and splits
  each bar's return into an overnight leg (earned by the *old* position) and an
  intraday leg (earned by the *newly filled* position). You physically cannot
  reference a price you used in the decision.
- **Proof:** `tests/test_lookahead.py` truncates and perturbs *future* bars and
  asserts past P&L is byte-identical. It also flips on a deliberately-buggy
  `allow_lookahead` path and asserts it *does* inflate returns — so the guard is
  provably sensitive to the thing it guards.
- **Demo:** the +19.2% "fantasy" return above is exactly this bug, quantified.

### 2. Survivorship bias — *only backtesting the winners*

If your universe is "S&P 500 companies that still trade today," you've silently
deleted every company that went bankrupt or got delisted. Your backtest never
buys Lehman.

- **Mechanism:** `data/storage/postgres_store.py` stores a point-in-time symbol
  table with `listed_on` / `delisted_on`, and `universe_asof(date)` returns the
  universe **as it existed on that date**, including names now dead.
- **Proof:** a symbol delisted in 2021 appears in the 2021 universe query and is
  absent from the 2023 one (see the Postgres section below).
- **Honesty note:** yfinance itself is survivorship-biased (delisted tickers
  return nothing). This is *why* the Polygon path and the point-in-time table
  exist — the tool can't fix bad source data, only refuse to pretend.

### 3. Corporate-action / adjustment bias — *the split that never crashed*

Two symmetric traps. Compute returns from **raw** close and a 2:1 split looks
like a −50% crash. Compute price *levels* from **back-adjusted** close and you
smear a future split back through history — a trader in 2015 never saw the
2020-adjusted price, so an absolute-level signal ("buy below \$50") on adjusted
data is a subtle look-ahead.

- **Mechanism:** every bar carries **both** `close` and `adj_close`
  (`data/bars.py`). `data/corporate_actions.py` exposes the adjustment factor
  and a split/dividend-consistent total-return series, so P&L is driven by
  returns (adjustment-invariant) and level signals can use raw prices —
  each choice explicit rather than accidental.
- **Proof:** `tests/test_corporate_actions.py` shows the raw series has a
  spurious −50% day at the split while the total-return series does not.

### 4. Transaction costs — *the strategy that only works for free*

A strategy that rebalances daily pays the spread daily. Gross returns are a
fantasy; net returns are the business.

- **Mechanism:** `engine/costs.py` models proportional costs (commission_bps +
  slippage_bps) and non-proportional ones (per-share, fixed). Presets:
  `zero()`, `retail()` (commission-free but ~5bps slippage), `institutional()`.
- **Proof:** `tests/test_costs.py` asserts more cost ⇒ less money, **and** that
  a high-turnover mean-reversion strategy loses more of its gross return to the
  same cost model than buy-and-hold does.

### 5. Slippage / execution assumptions — *you are not the only trader*

You never fill at the printed price. Slippage is modeled as a symmetric
half-spread: you buy a touch above and sell a touch below the reference open.
The event-driven engine fills at the **next** open with this penalty and tracks
real cash and shares, so fractional-share and fee effects are exact rather than
assumed away.

### 6. Overfitting / the vectorized-vs-reality gap

The fast vectorized engine assumes proportional costs and continuous weights;
the event-driven engine assumes nothing. Where they *disagree* is where a
research shortcut has snuck in an assumption. Keeping both and asserting parity
turns "did my fast backtest cheat?" into a unit test.

---

## Architecture

```
                 ┌────────────┐   DataProvider ABC
   Polygon.io ──▶│  providers │◀── yfinance (keyless fallback)
                 └─────┬──────┘
                       │  validated OHLCV bars (raw + adjusted)
                       ▼
                 ┌────────────┐        ┌──────────────────────────┐
                 │  ingest    │───────▶│ Parquet lake (bars)      │
                 └────────────┘   └───▶│ Postgres (metadata+runs) │
                                       └──────────────────────────┘
                       │
        Strategy ─────▶│  target weights (causal)
     (SMA / MeanRev)   ▼
                 ┌─────────────────────────────┐
                 │  vectorized  ║  event-driven │  ← share one CostModel,
                 │   (fast)     ║  (accurate)   │    verified to agree
                 └──────────────╨──────────────┘
                       │  BacktestResult
                       ▼
                 metrics (Sharpe, Sortino, CAGR, maxDD, turnover)
```

Everything downstream of `providers` depends only on abstract interfaces
(`DataProvider`, `BarStore`, `Strategy`), so swapping Polygon for IEX, Parquet
for DuckDB, or adding a new strategy touches exactly one file.

---

## Quickstart

Requires [`uv`](https://github.com/astral-sh/uv) (Python env) and, optionally,
Docker (Postgres). No API key needed for the default path.

```bash
make install      # uv venv + editable install
make demo         # ingest real SPY data, run both engines, print the pitfall report
make test         # 20 tests: look-ahead, parity, costs, corporate actions, storage
```

### CLI

```bash
# Ingest daily bars into the Parquet lake (and Postgres if MARKETLAB_PG_DSN is set)
marketlab ingest --symbols SPY AAPL MSFT --start 2015-01-01 --end 2023-12-31

# Backtest a strategy through both engines with a realistic cost model
marketlab backtest --symbol SPY --strategy sma_crossover --fast 50 --slow 200 \
                   --engine both --cost retail
```

### Using Polygon instead of yfinance

```bash
cp .env.example .env      # then set POLYGON_API_KEY and MARKETLAB_PROVIDER=polygon
```

The Polygon provider (`data/providers/polygon.py`) pulls **both** adjusted and
raw aggregates, rate-limits for the free tier (5 req/min), and pages the cursor.

### Postgres layer (optional but wired)

```bash
make db-up        # docker compose: postgres:16
export MARKETLAB_PG_DSN=postgresql://marketlab:marketlab@localhost:5432/marketlab
marketlab ingest --symbols AAPL --start 2022-01-01 --end 2022-12-31
```

Postgres holds the survivorship-aware symbol table and a `backtest_runs` table
that records every run's strategy, params, engine, and metrics as JSON for
reproducibility. Point-in-time universe:

```python
from datetime import date
from marketlab.data.storage import PostgresStore
pg = PostgresStore(dsn)
pg.upsert_symbol("XYZ", listed_on=date(2020,1,1), delisted_on=date(2021,6,1))
pg.universe_asof(date(2021,1,1))   # ['XYZ']  -> alive
pg.universe_asof(date(2023,1,1))   # []        -> correctly gone
```

---

## The two engines

| | vectorized | event-driven |
|---|---|---|
| speed | fast (numpy, whole-array) | slower (bar-by-bar loop) |
| accounting | weights × returns, compounded legs | explicit cash + shares |
| costs | proportional only | proportional + per-share + fixed |
| use | research sweeps | the number you'd actually trade |

They implement the *same* execution model (fill at next open) so they can be
cross-checked. `tests/test_engine_parity.py` asserts they match to `1e-9` at
zero cost and stay within a fraction of a basis point under proportional costs.
The residual is documented, not swept away — it comes from costs being charged
against cash (event) vs. against return (vectorized), and from fixed weights
(vectorized) vs. fixed shares that drift between rebalances (event).

---

## Layout

```
src/marketlab/
  config.py                 env/.env settings
  data/
    bars.py                 canonical OHLCV schema + validation
    corporate_actions.py    adjustment factor, total-return series, split detection
    providers/              DataProvider ABC, polygon, yfinance
    storage/                BarStore ABC, parquet_store, postgres_store
    ingest.py               provider -> validated bars -> store(s)
  engine/
    costs.py                proportional + non-proportional cost model
    vectorized.py           fast engine (overnight/intraday decomposition)
    event_driven.py         accurate engine (cash + shares)
    types.py                BacktestResult
  strategies/               Strategy ABC, buy&hold, SMA crossover, mean reversion
  metrics/                  Sharpe, Sortino, CAGR, max drawdown, turnover
  backtest.py, cli.py       orchestration + CLI
tests/                      look-ahead, parity, costs, corporate actions, storage, metrics
scripts/demo.py             the end-to-end pitfall report
```

---

## Honest limitations (things a naive backtest also hides)

- **Single-instrument engines.** Both engines run one symbol at a time. A true
  cross-sectional portfolio (weight drift across many names, rebalancing bands)
  is the natural next step; the vectorized decomposition generalizes to a weight
  matrix, the event engine to a positions dict.
- **Daily bars only.** Intraday fills are modeled as a single next-open price
  with a slippage penalty — no order book, no partial fills, no market impact
  that scales with size.
- **No point-in-time fundamentals.** Only prices are point-in-time; this doesn't
  ingest as-reported financials, so it can't catch restatement look-ahead.
- **Costs are a model, not a fill log.** Real slippage is state-dependent
  (volatility, spread, ADV participation); the constant-bps model is a floor.

These are listed because the fastest way to spot someone who *hasn't* hit these
pitfalls firsthand is that they claim not to have any.
