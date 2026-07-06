"""Strategy interface.

A strategy maps a bar frame to a series of **target weights**, one per bar,
where the weight at bar t is the position you want to hold *going into the next
bar* -- decided using only information available at the close of bar t.

The single hard rule: ``target_weights`` must be *causal*. The value at index t
may depend on bars <= t but never on bars > t. Indicators built from rolling
windows of closes satisfy this automatically; the engine then adds the
execution delay (fill at next open). The look-ahead test enforces causality.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "abstract"

    @abstractmethod
    def target_weights(self, bars: pd.DataFrame) -> pd.Series:
        """Return target weights in [-1, 1] aligned to bars.index."""

    def params(self) -> dict:
        return {
            k: v for k, v in vars(self).items() if not k.startswith("_")
        }
