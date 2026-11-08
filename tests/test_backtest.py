import numpy as np
import pandas as pd

from qp.backtest import metrics as M
from qp.backtest.engine import run_backtest
from qp.config import RunConfig
from qp.signals.rules import Momentum


def _cfg(**over):
    base = {
        "name": "t",
        "data": {"symbols": ["UP", "DOWN"]},
        "portfolio": {"sizing": "fixed", "gross_leverage": 1.0, "rebalance": "daily",
                      "max_weight": 1.0},
        "costs": {"commission_bps": 0, "half_spread_bps": 0, "borrow_bps_annual": 0},
        "risk": {"max_drawdown_stop": 0.0, "max_gross": 5.0, "max_position": 1.0},
    }
    base.update(over)
    return RunConfig.model_validate(base)


def test_zero_signal_is_flat(trend_prices):
    scores = pd.DataFrame(0.0, index=trend_prices.index, columns=trend_prices.columns)
    res = run_backtest(trend_prices, scores, _cfg())
    assert res.returns.abs().max() < 1e-12
    assert abs(res.equity.iloc[-1] - 1.0) < 1e-9


def test_momentum_makes_money_on_a_clean_trend(trend_prices):
    scores = Momentum(lookback=20, skip=0, vol_scale=False).generate(trend_prices)
    res = run_backtest(trend_prices, scores, _cfg())
    # long the up-trend, short the down-trend -> positive
    assert res.metrics["cagr"] > 0
    assert res.metrics["sharpe"] > 0


def test_costs_reduce_return(trend_prices):
    scores = Momentum(lookback=20, skip=0, vol_scale=False).generate(trend_prices)
    free = run_backtest(trend_prices, scores, _cfg())
    costed = run_backtest(
        trend_prices, scores,
        _cfg(costs={"commission_bps": 5, "half_spread_bps": 10, "borrow_bps_annual": 100}),
    )
    assert costed.returns.sum() < free.returns.sum()


def test_no_lookahead_one_bar_lag(trend_prices):
    # a signal that is 1 on the last bar only must not affect that bar's return
    scores = pd.DataFrame(0.0, index=trend_prices.index, columns=trend_prices.columns)
    scores.iloc[-1] = 1.0
    res = run_backtest(trend_prices, scores, _cfg())
    assert abs(res.returns.iloc[-1]) < 1e-12  # position only held from t+1, which doesn't exist


def test_drawdown_stop_flattens(trend_prices):
    # force a losing book (short the up-trend), tight stop -> book gets flattened
    scores = pd.DataFrame({"UP": -1.0, "DOWN": 0.0}, index=trend_prices.index)
    res = run_backtest(trend_prices, scores, _cfg(risk={"max_drawdown_stop": 0.05,
                                                       "max_gross": 5.0, "max_position": 1.0}))
    # after the stop trips there are zero-return days (flat)
    assert (res.returns == 0).sum() > 10


def test_metrics_sharpe_sign():
    up = pd.Series(np.r_[[0.001] * 252])
    assert M.sharpe(up) > 0
    assert M.sharpe(-up) < 0
    assert M.drawdown(up).min() == 0.0
