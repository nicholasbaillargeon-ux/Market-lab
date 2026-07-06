"""Vectorized backtester -- fast, for research sweeps.

Execution model (the honest one): a target weight decided at the *close* of bar
t is filled at the *open* of bar t+1. This one-bar delay is the structural guard
against look-ahead bias -- you can never trade on a price you used to make the
decision.

The trick that makes a pure-numpy engine execute-at-open *exactly* (so it can be
verified against the event-driven engine) is to split each bar's return into two
pieces and hand them to the correct weight:

    overnight[t] = adj_open[t]  / adj_close[t-1] - 1   # earned by the OLD position
    intraday[t]  = adj_close[t] / adj_open[t]    - 1   # earned by the NEW position

At open[t] we rebalance from the weight held into the open (decided at close
t-2) to the new weight (decided at close t-1). The two legs and the rebalance
cost *compound* (they don't add), which is what makes this match the
event-driven engine exactly at zero cost:

    w_new[t] = w.shift(1)   # filled at this open, earns the intraday leg
    w_old[t] = w.shift(2)   # held overnight into this open, earns the overnight leg
    factor[t] = (1 + w_old*overnight) * (1 - turnover*rate) * (1 + w_new*intraday)
    turnover[t] = |w_new[t] - w_old[t]|

`allow_lookahead=True` shifts execution one bar earlier -- you fill at the open
of the very bar whose close produced the signal, which no one can do. For a
trend strategy that captures the breakout day itself and inflates returns. It is
never used in real backtests; the look-ahead test asserts the default is
future-blind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import CostModel
from .types import BacktestResult


def run_vectorized(
    bars: pd.DataFrame,
    weights: pd.Series,
    cost_model: CostModel | None = None,
    *,
    initial_cash: float = 100_000.0,
    allow_lookahead: bool = False,
) -> BacktestResult:
    cost_model = cost_model or CostModel.zero()
    idx = bars.index
    w = weights.reindex(idx).astype("float64").fillna(0.0).clip(-1.0, 1.0)

    factor = bars["adj_close"] / bars["close"]
    adj_open = bars["open"] * factor
    adj_close = bars["adj_close"]

    overnight = (adj_open / adj_close.shift(1) - 1.0).fillna(0.0)
    intraday = (adj_close / adj_open - 1.0).fillna(0.0)

    # Honest: fill at the open one bar after the signal (shift 1). Look-ahead:
    # fill at the open of the signal's own bar (shift 0) -- impossible in reality.
    exec_shift = 0 if allow_lookahead else 1
    w_new = w.shift(exec_shift).fillna(0.0)
    w_old = w.shift(exec_shift + 1).fillna(0.0)

    turnover = (w_new - w_old).abs()
    rebalance = 1.0 - turnover * cost_model.proportional_rate
    # Legs compound; the rebalance cost is charged at the open, between them.
    factor = (1.0 + w_old * overnight) * rebalance * (1.0 + w_new * intraday)
    port_ret = (factor - 1.0).fillna(0.0)

    equity = initial_cash * factor.cumprod()

    # Reconstruct a trade blotter from weight changes for reporting/parity.
    trades = _blotter(idx, w_new, w_old, equity, adj_open, cost_model, allow_lookahead)

    return BacktestResult(
        engine="vectorized",
        equity=equity.rename("equity"),
        returns=port_ret.rename("ret"),
        weights=w_new.rename("weight"),   # weight actually held through each bar
        trades=trades,
        initial_cash=initial_cash,
        cost_model_repr=repr(cost_model),
        meta={"allow_lookahead": allow_lookahead},
    )


def _blotter(idx, w_new, w_old, equity, price, cost_model, lookahead) -> pd.DataFrame:
    dw = (w_new - w_old).to_numpy()
    eq = equity.to_numpy()
    px = price.to_numpy()
    rows = []
    for i in range(len(idx)):
        if abs(dw[i]) < 1e-12:
            continue
        # notional traded ~ |Δweight| * equity at the rebalance
        base_eq = eq[i - 1] if i > 0 else eq[i]
        notional = abs(dw[i]) * base_eq
        qty = notional / px[i] if px[i] else 0.0
        rows.append({
            "ts": idx[i],
            "side": "BUY" if dw[i] > 0 else "SELL",
            "qty": qty,
            "price": px[i],
            "cost": cost_model.trade_cost(qty, px[i]),
        })
    return pd.DataFrame(rows, columns=["ts", "side", "qty", "price", "cost"])
