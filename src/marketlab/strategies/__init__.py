from .base import Strategy
from .library import BuyAndHold, MeanReversion, SMACrossover

REGISTRY = {
    s.name: s
    for s in (BuyAndHold, SMACrossover, MeanReversion)
}

__all__ = ["Strategy", "BuyAndHold", "SMACrossover", "MeanReversion", "REGISTRY"]
