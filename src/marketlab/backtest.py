"""High-level glue: load bars from a store, run a strategy through an engine,
compute metrics, optionally persist the run to Postgres."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import CostModel, run_event_driven, run_vectorized
from .engine.types import BacktestResult
from .metrics import summary
from .strategies import REGISTRY, Strategy

ENGINES = {"vectorized": run_vectorized, "event_driven": run_event_driven}

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
