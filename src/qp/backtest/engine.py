"""Vectorised multi-asset backtester.

Timeline convention (no look-ahead):
    * signal & target weights for bar ``t`` are computed from data up to close ``t``
    * those weights are *held over* ``t -> t+1``  (one-bar execution lag)
    * costs are charged on the turnover that moves yesterday's book to today's

Returns a :class:`BacktestResult` with the net-return series, weights actually
held, per-component P&L, and the metrics summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import RunConfig
from ..logging import get_logger
from ..risk.limits import apply_limits
from . import costs as cost_mod
from . import metrics as metric_mod
from .portfolio import build_weights

log = get_logger(__name__)


@dataclass
class BacktestResult:
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    pnl_components: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return metric_mod.equity_curve(self.returns)


def run_backtest(close: pd.DataFrame, scores: pd.DataFrame, cfg: RunConfig) -> BacktestResult:
    close = close.sort_index()
    asset_ret = close.pct_change()

    target_w = build_weights(scores, close, cfg.portfolio)
    target_w = apply_limits(target_w, cfg.risk)

    # execution lag: hold weights decided at t over t -> t+1
    held = target_w.shift(1).fillna(0.0)

    gross_ret = (held * asset_ret).sum(axis=1)
    tc = cost_mod.transaction_cost(target_w, cfg.costs)      # charged when we move the book
    bc = cost_mod.borrow_cost(held, cfg.costs)
    net_ret = gross_ret - tc - bc

    # drawdown kill-switch: once tripped, flatten until a new equity high
    net_ret, held = _apply_drawdown_stop(net_ret, held, cfg.risk.max_drawdown_stop)

    pnl = pd.DataFrame(
        {
            "gross": gross_ret,
            "transaction_cost": -tc,
            "borrow_cost": -bc,
            "net": net_ret,
        }
    )

    res = BacktestResult(
        returns=net_ret.rename(cfg.name),
        gross_returns=gross_ret.rename(f"{cfg.name}:gross"),
        weights=held,
        pnl_components=pnl,
    )
    res.metrics = metric_mod.summary(net_ret, held)
    log.info(
        "backtest [bold]%s[/bold] | Sharpe %.2f | CAGR %.1f%% | maxDD %.1f%% | turn/day %.2f",
        cfg.name,
        res.metrics["sharpe"],
        100 * res.metrics["cagr"],
        100 * res.metrics["max_drawdown"],
        res.metrics.get("avg_daily_turnover", float("nan")),
    )
    return res


def _apply_drawdown_stop(
    net_ret: pd.Series,
    held: pd.DataFrame,
    max_dd: float,
    cooldown: int = 21,
) -> tuple[pd.Series, pd.DataFrame]:
    """Flatten the book for ``cooldown`` trading days after equity drawdown
    breaches ``max_dd``, then resume. Peak resets on resume so the rule keys off
    the drawdown *since the last restart*, which is how a real risk desk runs it.
    """
    if max_dd <= 0:
        return net_ret, held
    eq = 1.0
    peak = 1.0
    cooling = 0
    out = net_ret.copy()
    mask = pd.Series(1.0, index=net_ret.index)
    for i, r in enumerate(net_ret.to_numpy()):
        if cooling > 0:
            out.iloc[i] = 0.0
            mask.iloc[i] = 0.0
            cooling -= 1
            if cooling == 0:
                eq = peak = 1.0  # fresh start after the cooldown
            continue
        eq *= 1 + r
        peak = max(peak, eq)
        if eq / peak - 1 <= -max_dd:
            cooling = cooldown
    return out, held.mul(mask, axis=0)
