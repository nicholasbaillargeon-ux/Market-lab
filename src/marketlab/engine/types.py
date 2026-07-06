"""Shared result types for both engines."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    engine: str
    equity: pd.Series          # portfolio value marked at each bar's close
    returns: pd.Series         # net per-bar portfolio returns
    weights: pd.Series         # realized net exposure actually held each bar (post-fill)
    trades: pd.DataFrame       # ts, side, qty, price, cost  (+ symbol for portfolios)
    initial_cash: float
    cost_model_repr: str = ""
    meta: dict = field(default_factory=dict)
    weights_frame: pd.DataFrame | None = None  # per-asset held weights (multi-asset only)

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1])

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_cash - 1.0

    @property
    def turnover(self) -> float:
        """Sum of |weight changes| -- how much trading the strategy did.

        For a portfolio this must count the drift-correcting rebalance across
        *every* asset, not just the net exposure, so the engine records the true
        figure in ``meta['turnover']``; fall back to the single-asset series.
        """
        if "turnover" in self.meta:
            return float(self.meta["turnover"])
        if self.weights_frame is not None:
            return float(self.weights_frame.diff().abs().sum().sum())
        return float(self.weights.diff().abs().sum())
