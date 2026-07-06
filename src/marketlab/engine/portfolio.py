"""Multi-asset backtester -- the same two engines, generalized to N instruments.

The single-instrument engines agree to machine precision because a long/flat
position on one asset never *drifts*: your weight is 0 or 1, so the "weight held
into the next bar" the fast vectorized engine assumes is exactly the weight the
share-tracking event engine realizes. The moment you hold two assets at
fractional weights that stops being true -- between rebalances each holding's
weight drifts with its own return, so a naive ``sum(weight * asset_return)``
vectorized portfolio silently disagrees with a real share-based book.

This module keeps the parity guarantee by modelling one specific, well-defined
portfolio: **rebalance to the target weights at every open.** Both engines model
*that* portfolio, so they still agree to numerical precision -- the multi-asset
version of the same verification the single-asset parity test buys us.

The vectorized math is the single-asset two-leg decomposition, done per asset
and summed across the book, with one extra term that is identically zero in the
single-asset long/flat case (which is why this reduces exactly to
``run_vectorized`` there):

    a[t]   = target weights rebalanced to at open t         (W.shift(exec_shift))
    L_in[t]  = 1 + Σ_j a[t,j] * intraday[t,j]               # open t -> close t
    c[t,j]   = a[t,j] * (1 + intraday[t,j]) / L_in[t]        # weights DRIFTED to close t
    L_ov[t]  = 1 + Σ_j c[t-1,j] * overnight[t,j]            # close t-1 -> open t
    turn[t]  = Σ_j |a[t,j] - c[t-1,j]|                       # rebalanced at open t
    factor[t] = L_ov[t] * (1 - turn[t]*rate) * L_in[t]

``c`` -- the post-intraday drifted weights -- is the whole trick: the position
held *overnight* into bar t is the one you drifted to by the close of t-1, not
the clean target you set at t-1's open. Carry that and the fast engine
reproduces the share-based path exactly (zero cost); collapse it (single asset,
weights in {0,1}) and ``c == a`` and every formula above becomes the existing
single-asset one.

``allow_lookahead=True`` shifts execution one bar earlier, exactly as in the
single-asset engine -- you rebalance on the open of the very bar whose close
produced the weights, which nobody can do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import CostModel
from .types import BacktestResult

__all__ = ["align_bars", "run_vectorized_portfolio", "run_event_driven_portfolio"]


def align_bars(
    bars: dict[str, pd.DataFrame], symbols: list[str] | None = None
) -> tuple[list[str], pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Align a dict of per-symbol bar frames onto a common trading calendar.

    Uses the *intersection* of every symbol's index -- the dates on which the
    whole book actually traded together -- so there are no synthetic prices or
    ambiguous "was it listed yet?" gaps to bias the backtest. Returns the symbol
    order plus adjusted open/close arrays shaped ``(T, J)``.
    """
    symbols = list(symbols) if symbols is not None else sorted(bars)
    missing = [s for s in symbols if s not in bars or bars[s].empty]
    if missing:
        raise ValueError(f"no bars for symbols: {missing}")

    idx = None
    for s in symbols:
        idx = bars[s].index if idx is None else idx.intersection(bars[s].index)
    idx = idx.sort_values()
    if len(idx) == 0:
        raise ValueError("symbols share no common trading dates")

    adj_open = np.empty((len(idx), len(symbols)))
    adj_close = np.empty((len(idx), len(symbols)))
    for j, s in enumerate(symbols):
        b = bars[s].loc[idx]
        factor = (b["adj_close"] / b["close"]).to_numpy()
        adj_open[:, j] = b["open"].to_numpy() * factor
        adj_close[:, j] = b["adj_close"].to_numpy()
    return symbols, idx, adj_open, adj_close


def _prep_weights(
    weights: pd.DataFrame, symbols: list[str], idx: pd.DatetimeIndex
) -> np.ndarray:
    """Target weights as a clipped, gap-free ``(T, J)`` array aligned to idx."""
    w = weights.reindex(index=idx, columns=symbols)
    return w.astype("float64").fillna(0.0).clip(-1.0, 1.0).to_numpy()


def run_vectorized_portfolio(
    bars: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    cost_model: CostModel | None = None,
    *,
    initial_cash: float = 100_000.0,
    allow_lookahead: bool = False,
) -> BacktestResult:
    cost_model = cost_model or CostModel.zero()
    symbols = list(weights.columns)
    symbols, idx, adj_open, adj_close = align_bars(bars, symbols)
    n = len(idx)

    W = _prep_weights(weights, symbols, idx)

    prev_close = np.vstack([adj_close[:1], adj_close[:-1]])       # adj_close[t-1]
    overnight = adj_open / prev_close - 1.0
    overnight[0] = 0.0
    intraday = adj_close / adj_open - 1.0

    # a[t]: weights rebalanced to at open t (target decided at close t-1).
    exec_shift = 0 if allow_lookahead else 1
    a = np.zeros((n, len(symbols)))
    if exec_shift < n:
        a[exec_shift:] = W[: n - exec_shift]

    # Intraday leg and the weights it drifts each target to by the close.
    l_in = 1.0 + (a * intraday).sum(axis=1)
    c = a * (1.0 + intraday) / l_in[:, None]                     # drifted close-of-bar weights

    c_prev = np.vstack([np.zeros((1, len(symbols))), c[:-1]])    # held overnight into bar t
    l_ov = 1.0 + (c_prev * overnight).sum(axis=1)

    # Pre-rebalance weights *at the open* -- c_prev drifted one more leg by the
    # overnight move. This, not the close-of-t-1 weight, is what you actually
    # trade away from, so both the cost and the reported turnover use it. (For a
    # single asset at weight 0/1 it collapses back to c_prev, matching the
    # single-asset engine exactly.)
    open_w = c_prev * (1.0 + overnight) / l_ov[:, None]
    turnover = np.abs(a - open_w).sum(axis=1)                    # rebalance at each open
    rebalance = 1.0 - turnover * cost_model.proportional_rate

    factor = l_ov * rebalance * l_in
    equity = initial_cash * np.cumprod(factor)
    port_ret = factor - 1.0

    trades = _blotter(idx, symbols, a, open_w, equity, adj_open, cost_model)

    held = pd.DataFrame(a, index=idx, columns=symbols)
    return BacktestResult(
        engine="vectorized_portfolio",
        equity=pd.Series(equity, index=idx, name="equity"),
        returns=pd.Series(port_ret, index=idx, name="ret"),
        weights=pd.Series(a.sum(axis=1), index=idx, name="net_exposure"),
        trades=trades,
        initial_cash=initial_cash,
        cost_model_repr=repr(cost_model),
        meta={"allow_lookahead": allow_lookahead, "symbols": symbols,
              "turnover": float(turnover.sum())},
        weights_frame=held,
    )


def _blotter(idx, symbols, a, open_w, equity, adj_open, cost_model) -> pd.DataFrame:
    """Reconstruct a per-asset trade blotter from the per-open rebalances."""
    dw = a - open_w
    eq = equity
    rows = []
    for t in range(len(idx)):
        base_eq = eq[t - 1] if t > 0 else float(equity[0])
        for j, sym in enumerate(symbols):
            if abs(dw[t, j]) < 1e-12:
                continue
            px = adj_open[t, j]
            notional = abs(dw[t, j]) * base_eq
            qty = notional / px if px else 0.0
            rows.append({
                "ts": idx[t], "symbol": sym,
                "side": "BUY" if dw[t, j] > 0 else "SELL",
                "qty": qty, "price": px,
                "cost": cost_model.trade_cost(qty, px),
            })
    return pd.DataFrame(rows, columns=["ts", "symbol", "side", "qty", "price", "cost"])


def run_event_driven_portfolio(
    bars: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    cost_model: CostModel | None = None,
    *,
    initial_cash: float = 100_000.0,
) -> BacktestResult:
    """Share-based reference book. Rebalances every asset to its target weight at
    each open, tracking explicit cash and per-asset share counts -- the honest
    path the vectorized engine is checked against."""
    cost_model = cost_model or CostModel.zero()
    symbols = list(weights.columns)
    symbols, idx, adj_open, adj_close = align_bars(bars, symbols)
    n, j = len(idx), len(symbols)

    W = _prep_weights(weights, symbols, idx)

    cash = float(initial_cash)
    units = np.zeros(j)
    equity = np.empty(n)
    held_w = np.zeros((n, j))
    trades: list[dict] = []
    gross_turnover = 0.0

    for t in range(n):
        if t >= 1:
            target_w = W[t - 1]                          # decided at close t-1
            price = adj_open[t]                          # ... filled at open t
            equity_at_open = cash + float(units @ price)
            target_units = np.where(price > 0, target_w * equity_at_open / price, 0.0)
            delta = target_units - units
            if equity_at_open:                           # fraction of book traded
                pretrade_w = units * price / equity_at_open
                gross_turnover += float(np.abs(target_w - pretrade_w).sum())
            for k in range(j):
                if abs(delta[k]) < 1e-12:
                    continue
                cost = cost_model.trade_cost(delta[k], price[k])
                cash -= delta[k] * price[k] + cost
                trades.append({
                    "ts": idx[t], "symbol": symbols[k],
                    "side": "BUY" if delta[k] > 0 else "SELL",
                    "qty": abs(delta[k]), "price": price[k], "cost": cost,
                })
            units = target_units

        mkt = units * adj_close[t]
        equity[t] = cash + float(mkt.sum())
        if equity[t]:
            held_w[t] = mkt / equity[t]

    equity_s = pd.Series(equity, index=idx, name="equity")
    held = pd.DataFrame(held_w, index=idx, columns=symbols)
    return BacktestResult(
        engine="event_driven_portfolio",
        equity=equity_s,
        returns=equity_s.pct_change().fillna(0.0).rename("ret"),
        weights=pd.Series(held_w.sum(axis=1), index=idx, name="net_exposure"),
        trades=pd.DataFrame(trades, columns=["ts", "symbol", "side", "qty", "price", "cost"]),
        initial_cash=float(initial_cash),
        cost_model_repr=repr(cost_model),
        meta={"symbols": symbols, "turnover": gross_turnover},
        weights_frame=held,
    )
