"""Transaction and financing cost model."""
from __future__ import annotations

import pandas as pd

from ..config import CostsCfg


def turnover(weights: pd.DataFrame) -> pd.Series:
    """L1 change in target weights between consecutive bars."""
    return weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))


def transaction_cost(weights: pd.DataFrame, cfg: CostsCfg) -> pd.Series:
    bps = cfg.commission_bps + cfg.half_spread_bps + cfg.slippage_bps
    return turnover(weights) * bps / 1e4


def borrow_cost(weights: pd.DataFrame, cfg: CostsCfg) -> pd.Series:
    short_gross = weights.clip(upper=0).abs().sum(axis=1)
    return short_gross * (cfg.borrow_bps_annual / 1e4) / 252
