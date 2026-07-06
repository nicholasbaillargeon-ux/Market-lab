"""Costs must bite, and bite harder on high-turnover strategies -- the whole
reason a gross backtest overstates a real strategy."""
from __future__ import annotations

from marketlab.engine import run_event_driven, run_vectorized
from marketlab.engine.costs import CostModel
from marketlab.strategies import BuyAndHold, MeanReversion


def test_more_cost_less_money(synth_bars):
    strat = MeanReversion(lookback=20, entry_z=1.0)
    w = strat.target_weights(synth_bars)
    free = run_event_driven(synth_bars, w, CostModel.zero()).total_return
    cheap = run_event_driven(synth_bars, w, CostModel.retail()).total_return
    dear = run_event_driven(
        synth_bars, w, CostModel(commission_bps=0, slippage_bps=50)
    ).total_return
    assert free > cheap > dear


def test_costs_scale_with_turnover(synth_bars):
    """Same cost model: the high-turnover strategy should lose more of its gross
    return to costs than buy & hold."""
    cost = CostModel.retail()

    def cost_drag(strat):
        w = strat.target_weights(synth_bars)
        gross = run_vectorized(synth_bars, w, CostModel.zero()).total_return
        net = run_vectorized(synth_bars, w, cost).total_return
        return gross - net

    drag_bh = cost_drag(BuyAndHold())
    drag_mr = cost_drag(MeanReversion(lookback=10, entry_z=0.5))
    assert drag_mr > drag_bh


def test_trade_cost_components():
    m = CostModel(commission_bps=1.0, slippage_bps=4.0,
                  commission_per_share=0.01, fixed_per_trade=1.0)
    # 100 shares @ $50 = $5000 notional.
    # proportional: (1+4)bps * 5000 = 2.5 ; per-share: 0.01*100 = 1.0 ; fixed: 1.0
    assert abs(m.trade_cost(100, 50.0) - (2.5 + 1.0 + 1.0)) < 1e-12
    assert m.trade_cost(0, 50.0) == 0.0
