"""Pre-trade weight limits (the drawdown kill-switch lives in the engine)."""
from __future__ import annotations

import pandas as pd

from ..config import RiskCfg


def apply_limits(weights: pd.DataFrame, cfg: RiskCfg) -> pd.DataFrame:
    w = weights.clip(lower=-cfg.max_position, upper=cfg.max_position)
    gross = w.abs().sum(axis=1).replace(0, float("nan"))
    over = gross > cfg.max_gross
    if over.any():
        w.loc[over] = w.loc[over].div(gross[over], axis=0) * cfg.max_gross
    return w.fillna(0.0)
