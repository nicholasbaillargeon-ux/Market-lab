"""The headline test: prove the pipeline cannot see the future.

Look-ahead bias is the #1 reason backtests lie. We attack it two ways:

1. The strategy must be *causal*: weights up to bar t may not change when later
   bars change (or are removed).
2. The engine must not let future bars affect past P&L.

We also confirm the deliberately-buggy `allow_lookahead=True` path *does* leak,
so the test is actually sensitive to the thing it claims to guard.
"""
from __future__ import annotations

import numpy as np

from marketlab.engine import run_event_driven, run_vectorized
from marketlab.engine.costs import CostModel
from marketlab.strategies import MeanReversion, SMACrossover


def test_strategy_is_causal_under_truncation(synth_bars):
    strat = SMACrossover(fast=10, slow=50)
    full = strat.target_weights(synth_bars)
    k = 300
    truncated = strat.target_weights(synth_bars.iloc[:k])
    # Weights on the shared prefix must be identical: no dependence on bars > t.
    np.testing.assert_array_equal(full.iloc[:k].to_numpy(), truncated.to_numpy())


def test_strategy_causal_under_future_perturbation(synth_bars):
    strat = MeanReversion(lookback=20, entry_z=1.0)
    base = strat.target_weights(synth_bars)

    t = 250
    tampered = synth_bars.copy()
    tampered.iloc[t + 1:, :] *= 5.0  # blow up every FUTURE bar
    perturbed = strat.target_weights(tampered)

    np.testing.assert_array_equal(
        base.iloc[: t + 1].to_numpy(), perturbed.iloc[: t + 1].to_numpy()
    )


def test_engine_pnl_ignores_future(synth_bars):
    strat = SMACrossover(fast=10, slow=50)
    cost = CostModel.retail()
    t = 250

    for engine in (run_vectorized, run_event_driven):
        w = strat.target_weights(synth_bars)
        base = engine(synth_bars, w, cost)

        tampered = synth_bars.copy()
        tampered.iloc[t + 1:, :] *= 3.0
        w2 = strat.target_weights(tampered)
        pert = engine(tampered, w2, cost)

        # Equity up to and including bar t is unchanged by future tampering.
        np.testing.assert_allclose(
            base.equity.iloc[: t + 1].to_numpy(),
            pert.equity.iloc[: t + 1].to_numpy(),
            rtol=1e-9, atol=1e-6,
        )


def test_lookahead_flag_actually_leaks(synth_bars):
    """Sanity check: the buggy path must inflate returns, else our guard proves
    nothing. Filling at the open of the signal's own bar lets a trend strategy
    capture the breakout day it should have missed."""
    strat = SMACrossover(fast=10, slow=50)
    w = strat.target_weights(synth_bars)
    honest = run_vectorized(synth_bars, w, CostModel.zero())
    cheating = run_vectorized(synth_bars, w, CostModel.zero(), allow_lookahead=True)
    assert cheating.total_return > honest.total_return
