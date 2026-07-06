"""High-level glue: load bars from a store, run a strategy through an engine,
compute metrics, optionally persist the run to Postgres."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import (
    CostModel,
    run_event_driven,
    run_event_driven_portfolio,
    run_vectorized,
    run_vectorized_portfolio,
)
from .engine.types import BacktestResult
from .metrics import summary
from .strategies import PORTFOLIO_REGISTRY, REGISTRY, PortfolioStrategy, Strategy

ENGINES = {"vectorized": run_vectorized, "event_driven": run_event_driven}
PORTFOLIO_ENGINES = {
    "vectorized": run_vectorized_portfolio,
    "event_driven": run_event_driven_portfolio,
}

COST_PRESETS = {
    "zero": CostModel.zero,
    "retail": CostModel.retail,
    "institutional": CostModel.institutional,
}


@dataclass
class BacktestReport:
    result: BacktestResult
    metrics: dict


def resolve_strategy(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


def run_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    *,
    engine: str = "event_driven",
    cost_model: CostModel | None = None,
    initial_cash: float = 100_000.0,
) -> BacktestReport:
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; have {sorted(ENGINES)}")
    weights = strategy.target_weights(bars)
    result = ENGINES[engine](
        bars, weights, cost_model or CostModel.zero(), initial_cash=initial_cash
    )
    return BacktestReport(result=result, metrics=summary(result))


def resolve_portfolio_strategy(name: str, **params) -> PortfolioStrategy:
    if name not in PORTFOLIO_REGISTRY:
        raise ValueError(
            f"unknown portfolio strategy {name!r}; have {sorted(PORTFOLIO_REGISTRY)}"
        )
    return PORTFOLIO_REGISTRY[name](**params)


def run_portfolio_backtest(
    bars: dict[str, pd.DataFrame],
    strategy: PortfolioStrategy,
    *,
    engine: str = "event_driven",
    cost_model: CostModel | None = None,
    initial_cash: float = 100_000.0,
) -> BacktestReport:
    if engine not in PORTFOLIO_ENGINES:
        raise ValueError(
            f"unknown engine {engine!r}; have {sorted(PORTFOLIO_ENGINES)}"
        )
    weights = strategy.target_weights(bars)
    result = PORTFOLIO_ENGINES[engine](
        bars, weights, cost_model or CostModel.zero(), initial_cash=initial_cash
    )
    return BacktestReport(result=result, metrics=summary(result))
