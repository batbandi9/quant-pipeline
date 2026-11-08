"""Turn signal scores into leverage- and risk-constrained target weights.

Sizing schemes
--------------
fixed        weights proportional to score, scaled to ``gross_leverage``
vol_target   scale the whole book each day so the *ex-ante* portfolio vol
             (from a rolling covariance) tracks ``target_vol``
risk_parity  inverse-vol weights within the sign pattern of the score, then
             vol-targeted like above

All schemes then apply: per-name cap, optional long-only projection,
gross-leverage renormalisation, and the configured rebalance frequency.
The ex-ante vol estimate uses ``w' Σ w`` with a rolling sample covariance, so it
accounts for diversification and long/short netting -- not a sum of |w_i σ_i|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PortfolioCfg


def _rebalance_mask(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    if freq == "daily":
        return pd.Series(True, index=index)
    if freq == "weekly":
        wk = index.isocalendar().week.to_numpy()
        return pd.Series(np.r_[True, wk[1:] != wk[:-1]], index=index)
    if freq == "monthly":
        m = index.month.to_numpy()
        return pd.Series(np.r_[True, m[1:] != m[:-1]], index=index)
    raise ValueError(f"unknown rebalance freq {freq!r}")


def ex_ante_vol(weights: pd.DataFrame, asset_ret: pd.DataFrame, lookback: int) -> pd.Series:
    """Annualised sqrt(w' Σ w) with a rolling sample covariance."""
    rets = asset_ret.fillna(0.0)
    cov = rets.rolling(lookback).cov()
    out = pd.Series(index=weights.index, dtype=float)
    syms = list(weights.columns)
    for dt in weights.index:
        try:
            sigma = cov.loc[dt].reindex(index=syms, columns=syms).to_numpy()
        except KeyError:
            continue
        if np.isnan(sigma).any():
            continue
        w = weights.loc[dt, syms].to_numpy()
        var = float(w @ sigma @ w)
        out.loc[dt] = np.sqrt(max(var, 0.0)) * np.sqrt(252)
    return out


def build_weights(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    cfg: PortfolioCfg,
) -> pd.DataFrame:
    scores = scores.reindex_like(close).fillna(0.0)
    asset_ret = close.pct_change()
    asset_vol = (
        np.log(close).diff().rolling(cfg.vol_lookback).std() * np.sqrt(252)
    ).clip(lower=1e-3)

    if cfg.sizing == "risk_parity":
        raw = np.sign(scores) / asset_vol
    else:  # fixed | vol_target
        raw = scores

    raw = raw.clip(lower=-cfg.max_weight, upper=cfg.max_weight)
    if cfg.long_only:
        raw = raw.clip(lower=0.0)

    gross = raw.abs().sum(axis=1).replace(0, np.nan)
    w = raw.div(gross, axis=0).fillna(0.0) * cfg.gross_leverage

    if cfg.sizing in ("vol_target", "risk_parity"):
        pv = ex_ante_vol(w, asset_ret, cfg.vol_lookback)
        k = (cfg.target_vol / pv).clip(upper=cfg.max_leverage_scale)
        k = k.reindex(w.index).ffill().fillna(0.0)
        w = w.mul(k, axis=0)

        w = w.clip(lower=-cfg.max_weight, upper=cfg.max_weight)
        gross = w.abs().sum(axis=1).replace(0, np.nan)
        over = gross > max(cfg.gross_leverage, 1.0) * cfg.max_leverage_scale
        if over.any():
            w.loc[over] = w.loc[over].div(gross[over], axis=0) * (
                max(cfg.gross_leverage, 1.0) * cfg.max_leverage_scale
            )

    reb = _rebalance_mask(close.index, cfg.rebalance)
    w = w.where(reb, np.nan).ffill().fillna(0.0)
    return w
