# Quickstart walkthrough

Run these from the repo root after `pip install -e .`.

## 1. Look at the data

```python
from qp.data import load_prices, to_wide
from qp.config import DataCfg

df = load_prices(DataCfg(source="synthetic", symbols=["AAA","BBB","CCC","DDD"],
                         start="2015-01-01", end="2023-12-31"))
close = to_wide(df, "close")
close.plot(logy=True)
```

## 2. Build features

```python
from qp.features import technical as tech
mom  = tech.momentum(close, n=60, skip=5)
rsi  = tech.rsi(close, 14)
tech.compute("zscore_20", close).tail()   # token-resolved
```

## 3. Generate a signal

```python
from qp.signals.rules import CrossSectionalMomentum
scores = CrossSectionalMomentum(lookback=60, skip=5).generate(close)
```

## 4. Backtest with costs

```python
from qp.config import RunConfig
from qp.backtest import run_backtest

cfg = RunConfig.model_validate({
    "name": "nb", "data": {"symbols": list(close.columns)},
    "portfolio": {"sizing": "vol_target", "target_vol": 0.1},
    "costs": {"commission_bps": 1, "half_spread_bps": 2, "borrow_bps_annual": 50},
})
res = run_backtest(close, scores, cfg)
res.metrics            # dict: sharpe, cagr, max_drawdown, turnover, ...
res.equity.plot()
```

## 5. Compare a rule vs an ML model

```bash
qp walkforward -c configs/demo.yaml
```

## 6. Step the paper broker

```python
from qp.live.loop import run_once
rec = run_once(cfg)      # rebalances a JSON-persisted paper account
rec["broker"]            # {'equity': ..., 'positions': {...}, 'n_fills': ...}
```

## 7. Next: a real power-market signal

- `data.source: csv`, drop ENTSO-E day-ahead price CSVs in `data/csv/`
- new `Signal` subclass: features = load forecast, wind/solar forecast, gas price;
  target = day-ahead → intraday spread
- benchmark it against `CrossSectionalMomentum` in `qp walkforward`
