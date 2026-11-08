"""Rule-based alpha models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import technical as tech
from ..features.pipeline import cross_sectional_zscore
from .base import Signal


class Momentum(Signal):
    """Time-series momentum: long winners, short losers, per symbol."""

    def __init__(self, lookback: int = 120, skip: int = 5, vol_scale: bool = True):
        self.lookback = lookback
        self.skip = skip
        self.vol_scale = vol_scale

    def generate(self, close: pd.DataFrame) -> pd.DataFrame:
        mom = tech.momentum(close, n=self.lookback, skip=self.skip)
        score = np.tanh(mom / mom.abs().rolling(252, min_periods=60).mean())
        if self.vol_scale:
            vol = tech.realized_vol(close, n=20).clip(lower=1e-4)
            score = score / vol * vol.mean().mean()
        return score.clip(-1, 1)


class MeanReversion(Signal):
    """Short-horizon reversal: fade stretched z-scores."""

    def __init__(self, lookback: int = 5, entry_z: float = 1.0):
        self.lookback = lookback
        self.entry_z = entry_z

    def generate(self, close: pd.DataFrame) -> pd.DataFrame:
        z = tech.zscore(close, n=self.lookback)
        return (-z / (2 * self.entry_z)).clip(-1, 1)


class CrossSectionalMomentum(Signal):
    """Rank symbols by trailing return; hold a market-neutral long/short book."""

    market_neutral = True

    def __init__(self, lookback: int = 60, skip: int = 5):
        self.lookback = lookback
        self.skip = skip

    def generate(self, close: pd.DataFrame) -> pd.DataFrame:
        mom = tech.momentum(close, n=self.lookback, skip=self.skip)
        z = cross_sectional_zscore(mom)
        # scale so each row's gross exposure is ~1 before portfolio sizing
        gross = z.abs().sum(axis=1).replace(0, np.nan)
        return z.div(gross, axis=0).fillna(0.0)
