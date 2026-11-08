"""Performance and risk metrics from a daily net-return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def cagr(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    years = len(returns) / TRADING_DAYS
    if years <= 0 or eq.iloc[-1] <= 0:
        return float("nan")
    return eq.iloc[-1] ** (1 / years) - 1


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    sd = ex.std()
    return float("nan") if sd == 0 else np.sqrt(TRADING_DAYS) * ex.mean() / sd


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    downside = ex[ex < 0].std()
    return float("nan") if not downside else np.sqrt(TRADING_DAYS) * ex.mean() / downside


def summary(returns: pd.Series, weights: pd.DataFrame | None = None) -> dict[str, float]:
    returns = returns.dropna()
    dd = drawdown(returns)
    out = {
        "n_days": int(len(returns)),
        "cagr": cagr(returns),
        "ann_vol": float(returns.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": float(dd.min()),
        "calmar": float(cagr(returns) / abs(dd.min())) if dd.min() < 0 else float("nan"),
        "hit_rate": float((returns > 0).mean()),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "var_95": float(returns.quantile(0.05)),
        "cvar_95": float(returns[returns <= returns.quantile(0.05)].mean()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
    }
    if weights is not None:
        w = weights.reindex(returns.index).fillna(0.0)
        out["avg_gross_exposure"] = float(w.abs().sum(axis=1).mean())
        out["avg_net_exposure"] = float(w.sum(axis=1).mean())
        out["avg_daily_turnover"] = float(w.diff().abs().sum(axis=1).mean())
        out["time_in_market"] = float((w.abs().sum(axis=1) > 1e-6).mean())
    return out
