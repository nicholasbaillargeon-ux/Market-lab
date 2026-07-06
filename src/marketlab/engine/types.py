"""Shared result types for both engines."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    engine: str
    equity: pd.Series          # portfolio value marked at each bar's close
    returns: pd.Series         # net per-bar portfolio returns
    weights: pd.Series         # realized weight actually held each bar (post-fill)
    trades: pd.DataFrame       # ts, side, qty, price, cost
    initial_cash: float
    cost_model_repr: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1])

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_cash - 1.0

    @property
    def turnover(self) -> float:
        """Sum of |weight changes| -- how much trading the strategy did."""
        return float(self.weights.diff().abs().sum())
