"""Event-driven backtester -- the accurate reference.

Walks bar by bar with explicit cash and share holdings, no vectorization
shortcuts. On each bar it (1) fills the order decided at the previous close, at
*this* bar's open, paying the full cost model (including per-share and fixed
fees the vectorized engine can only approximate), then (2) marks the portfolio
to market at the close.

Because it tracks cash and shares directly, it naturally captures things the
vectorized engine glosses over: non-proportional fees, and the fact that between
rebalances you hold a fixed *share count* (whose weight drifts) rather than a
fixed weight. For a single instrument going long/flat those effects vanish, so
the two engines agree to numerical precision -- which is exactly what the parity
test checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import CostModel
from .types import BacktestResult


def run_event_driven(
    bars: pd.DataFrame,
    weights: pd.Series,
    cost_model: CostModel | None = None,
    *,
    initial_cash: float = 100_000.0,
) -> BacktestResult:
    cost_model = cost_model or CostModel.zero()
    idx = bars.index
    n = len(idx)

    w = weights.reindex(idx).astype("float64").fillna(0.0).clip(-1.0, 1.0).to_numpy()
    factor = (bars["adj_close"] / bars["close"]).to_numpy()
    adj_open = (bars["open"].to_numpy() * factor)
    adj_close = bars["adj_close"].to_numpy()

    cash = float(initial_cash)
    units = 0.0
    prev_target = 0.0                               # start flat
    equity = np.empty(n)
    held_w = np.zeros(n)
    trades: list[dict] = []

    for t in range(n):
        if t >= 1:
            target_w = w[t - 1]                     # decided at close t-1
            price = adj_open[t]                     # ... filled at open t
            # Only trade when the target actually changes. Otherwise hold the
            # existing shares (their weight drifts) -- rebalancing to an exact
            # weight every bar would churn spurious micro-trades and costs.
            if abs(target_w - prev_target) > 1e-12:
                equity_at_open = cash + units * price
                target_units = target_w * equity_at_open / price if price else 0.0
                delta = target_units - units
                prev_target = target_w
                cost = cost_model.trade_cost(delta, price)
                cash -= delta * price               # buy (delta>0) spends cash
                cash -= cost
                units += delta
                trades.append({
                    "ts": idx[t],
                    "side": "BUY" if delta > 0 else "SELL",
                    "qty": abs(delta),
                    "price": price,
                    "cost": cost,
                })

        equity[t] = cash + units * adj_close[t]
        held_w[t] = (units * adj_close[t] / equity[t]) if equity[t] else 0.0

    equity_s = pd.Series(equity, index=idx, name="equity")
    returns = equity_s.pct_change().fillna(0.0).rename("ret")

    return BacktestResult(
        engine="event_driven",
        equity=equity_s,
        returns=returns,
        weights=pd.Series(held_w, index=idx, name="weight"),
        trades=pd.DataFrame(trades, columns=["ts", "side", "qty", "price", "cost"]),
        initial_cash=float(initial_cash),
        cost_model_repr=repr(cost_model),
    )
