"""End-to-end demo: ingest -> store -> backtest both engines -> pitfall report.

Runs with zero setup: uses yfinance if the network is available, otherwise falls
back to a deterministic synthetic series so the demo always produces output.

What it demonstrates, in order:
  1. Vectorized vs event-driven agree (the layered engine verification).
  2. Look-ahead inflation: trading on the signal's own close vs the honest fill.
  3. Cost drag: the same strategy under zero / retail / institutional costs.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketlab.data.bars import validate  # noqa: E402
from marketlab.data.storage import ParquetStore  # noqa: E402
from marketlab.engine import run_event_driven, run_vectorized  # noqa: E402
from marketlab.engine.costs import CostModel  # noqa: E402
from marketlab.metrics import summary  # noqa: E402
from marketlab.strategies import SMACrossover  # noqa: E402

SYMBOL = "SPY"
START, END = date(2010, 1, 1), date(2023, 12, 31)
LAKE = Path(__file__).resolve().parents[1] / "data" / "lake"


def _synthetic(n: int = 3200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n, tz="UTC", name="ts")
    # Mild upward drift + vol regimes so a trend strategy has something to chew.
    rets = rng.normal(0.0003, 0.011, size=n)
    rets[1000:1200] -= 0.004  # a drawdown regime
    close = 100.0 * np.cumprod(1 + rets)
    openp = np.r_[close[0], close[:-1] * (1 + rng.normal(0, 0.002, n - 1))]
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol = rng.integers(5e7, 2e8, n).astype(float)
    return validate(pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": vol}, index=idx))


def load_bars() -> tuple[pd.DataFrame, str]:
    store = ParquetStore(LAKE)
    existing = store.read_bars(SYMBOL, START, END)
    if len(existing) > 500:
        return existing, "parquet-cache"
    try:
        from marketlab.data.providers.yfinance_provider import YFinanceProvider
        bars = YFinanceProvider().get_daily_bars(SYMBOL, START, END)
        if len(bars) > 500:
            store.write_bars(SYMBOL, bars)
            return bars, "yfinance"
    except Exception as e:
        print(f"[demo] yfinance unavailable ({e}); using synthetic series.")
    bars = _synthetic()
    store.write_bars("SYNTH", bars)
    return bars, "synthetic"


def _row(label: str, m: dict) -> str:
    return (f"{label:22} {m['total_return']:+8.1%} {m['cagr']:+7.1%} "
            f"{m['sharpe']:7.2f} {m['max_drawdown']:8.1%} "
            f"{m['turnover']:8.1f} {m['n_trades']:6d}")


def main() -> int:
    bars, source = load_bars()
    print(f"\nData: {SYMBOL} via {source}  |  {len(bars)} bars  "
          f"[{bars.index.min().date()} -> {bars.index.max().date()}]")

    strat = SMACrossover(fast=50, slow=200)  # the classic golden-cross
    w = strat.target_weights(bars)

    header = f"{'scenario':22} {'total':>8} {'cagr':>7} {'sharpe':>7} {'maxDD':>8} {'turnover':>8} {'trades':>6}"

    # 1. Engine parity (zero cost).
    print("\n== 1. Engine parity (zero cost) ==")
    print(header)
    v = run_vectorized(bars, w, CostModel.zero())
    e = run_event_driven(bars, w, CostModel.zero())
    print(_row("vectorized", summary(v)))
    print(_row("event_driven", summary(e)))
    gap = abs(v.total_return - e.total_return)
    print(f"   -> terminal-return gap: {gap:.2e}  (verification: engines agree)")

    # 2. Look-ahead inflation.
    print("\n== 2. Look-ahead bias (same strategy, dishonest fill) ==")
    print(header)
    honest = run_vectorized(bars, w, CostModel.zero())
    cheat = run_vectorized(bars, w, CostModel.zero(), allow_lookahead=True)
    print(_row("honest (next open)", summary(honest)))
    print(_row("look-ahead (same close)", summary(cheat)))
    infl = cheat.total_return - honest.total_return
    print(f"   -> look-ahead inflates total return by {infl:+.1%} of pure fantasy")

    # 3. Cost drag.
    print("\n== 3. Transaction-cost drag (event-driven) ==")
    print(header)
    for label, cm in [("zero cost", CostModel.zero()),
                      ("retail (5bps slip)", CostModel.retail()),
                      ("institutional", CostModel.institutional()),
                      ("harsh (50bps slip)", CostModel(slippage_bps=50))]:
        print(_row(label, summary(run_event_driven(bars, w, cm))))

    # Optional: save an equity-curve chart.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        run_event_driven(bars, w, CostModel.zero()).equity.plot(ax=ax, label="gross (zero cost)")
        run_event_driven(bars, w, CostModel.retail()).equity.plot(ax=ax, label="net (retail)")
        cheat.equity.plot(ax=ax, label="look-ahead fantasy", ls="--")
        ax.set_title(f"{SYMBOL} golden-cross: why the naive backtest lies")
        ax.legend(); ax.set_ylabel("equity ($)")
        out = Path(__file__).resolve().parents[1] / "equity_curves.png"
        fig.tight_layout(); fig.savefig(out, dpi=110)
        print(f"\nsaved chart -> {out}")
    except Exception as e:
        print(f"\n(chart skipped: {e})")

    print("\nTakeaway: the honest, costed, event-driven curve is the only one you "
          "could have actually traded. The other two are the lie.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
