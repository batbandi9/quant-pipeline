"""Signal contract.

``generate(close) -> DataFrame`` of **target scores** (index=date, columns=symbol).
Convention: scores are pre-sizing "desire to hold". They may be
    * directional in roughly [-1, 1] (e.g. momentum sign / tanh), or
    * cross-sectional weights that already sum to ~0 (market-neutral).
The portfolio layer turns scores into leverage-constrained weights.

Scores at row ``t`` must use only information available at the close of ``t``;
the backtester applies the one-bar execution lag.
"""
from __future__ import annotations

import abc

import pandas as pd


class Signal(abc.ABC):
    #: True if generate() already returns net-zero cross-sectional weights
    market_neutral: bool = False

    @abc.abstractmethod
    def generate(self, close: pd.DataFrame) -> pd.DataFrame:
        ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"
