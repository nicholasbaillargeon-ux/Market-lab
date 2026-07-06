"""Performance metrics. Daily bars -> annualized with 252 trading days."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = len(equity) / TRADING_DAYS
    if years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0


def annual_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std(ddof=0)
    if sd < 1e-12:  # constant returns -> Sharpe undefined; report 0
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd = np.sqrt((downside**2).mean()) if len(downside) else 0.0
    if dd < 1e-12:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def summary(result) -> dict:
    """Build the standard metric block from a BacktestResult."""
    eq, ret = result.equity, result.returns
    return {
        "engine": result.engine,
        "total_return": result.total_return,
        "cagr": cagr(eq),
        "ann_vol": annual_vol(ret),
        "sharpe": sharpe(ret),
        "sortino": sortino(ret),
        "max_drawdown": max_drawdown(eq),
        "turnover": result.turnover,
        "n_trades": int(len(result.trades)),
        "final_equity": result.final_equity,
    }
