# qp — a systematic trading pipeline

A full research-to-execution pipeline, built the way a small quant desk builds
one: config-driven, testable, no look-ahead, realistic costs, paper execution,
and an MCP layer so an agent can drive it.

> **Read this first.** Retail systematic trading is a near-zero-sum game against
> firms with better data, lower fees, and colocation. This repo is worth building
> for two reasons: (1) it is the portfolio piece that gets you onto an
> energy/commodity trading desk, and (2) it teaches market mechanics that
> compound. It is **not** an income strategy. The demo runs on synthetic data
> with *injected* structure so the machinery is visible; real markets are far
> noisier, and an in-sample equity curve means nothing.

## Quickstart

```bash
python -m pip install -e .          # core deps only (numpy/pandas/sklearn/matplotlib)
qp demo                             # end-to-end on synthetic data -> runs/demo/
qp backtest -c configs/demo.yaml
qp walkforward -c configs/demo.yaml # rule-based vs ML, side by side
qp paper -c configs/demo.yaml       # one live-loop step vs the paper broker
make test                           # 13 tests, synthetic data, no network
```

Optional extras: `pip install -e ".[ml,data,opt,mcp,dev]"`
(`ml` = torch/LSTM, `data` = yfinance, `opt` = cvxpy, `mcp` = the agent server).

## Pipeline stages

```
                 configs/*.yaml  (one file fully describes a run)
                        │
   data/     ───────────┼───────────  synthetic | csv | yfinance | stooq
   features/ ───────────┤             technical + calendar, all causal
   signals/  ───────────┤             momentum · mean-reversion · cross-sectional
                        │             · ml (sklearn) · sequence (LSTM)
   validation/ ─────────┤             walk-forward · purged K-fold + embargo
   backtest/  ──────────┤             vectorised, 1-bar lag, w'Σw vol targeting,
                        │             commission + spread + slippage + borrow
   risk/      ──────────┤             per-name cap · gross cap · drawdown stop
   execution/ ──────────┤             paper broker (default) · alpaca stub
   report/    ──────────┤             tearsheet.png + report.txt + metrics.json
   live/      ──────────┘             pull→features→signal→target→reconcile→orders
                        │
   mcp/server.py  ──────┘             the whole thing as MCP tools for an agent
```

## Layout

| Path | Purpose |
|---|---|
| `src/qp/config.py` | one pydantic model per stage; `RunConfig.from_yaml` |
| `src/qp/data/` | `DataSource` ABC + synthetic / CSV / yfinance / stooq; parquet cache |
| `src/qp/features/` | `technical.py` (token-resolved: `mom_60`, `rsi_14`…) + ML dataset builder |
| `src/qp/signals/` | `Signal` ABC + rules, `MLSignal` (walk-forward GBR), `SequenceSignal` (LSTM) |
| `src/qp/validation/` | `walk_forward`, `purged_kfold` with embargo |
| `src/qp/backtest/` | `engine.py`, `portfolio.py` (vol targeting via rolling cov), `costs.py`, `metrics.py` |
| `src/qp/risk/` | pre-trade weight limits (drawdown kill-switch is in the engine) |
| `src/qp/execution/` | `Broker` ABC, `PaperBroker` (JSON-persisted), `AlpacaBroker` stub |
| `src/qp/report/` | matplotlib tearsheet + text report |
| `src/qp/live/loop.py` | `run_once` (cron-safe) and `run_forever` |
| `src/qp/pipeline.py` | the orchestrator: config → prices → scores → backtest → report |
| `mcp/` | MCP server + design doc |
| `configs/` | `demo.yaml`, `ml_walkforward.yaml` |
| `tests/` | data/feature/backtest/execution tests on synthetic data |

## Design rules

- **No look-ahead.** Scores at bar `t` use data ≤ `t`; the engine holds them over
  `t → t+1`; ML/sequence models train walk-forward with an embargo covering the
  label horizon.
- **Costs are first-class.** Every backtest charges commission + half-spread +
  slippage on turnover and borrow on short exposure; the tearsheet decomposes
  P&L into gross vs cost drag.
- **Vol targeting is real.** The book is scaled each day by `target_vol / √(wᵀΣw)`
  with a rolling covariance — it accounts for diversification and long/short
  netting, not `Σ|wᵢσᵢ|`.
- **Determinism.** One `seed` in the config drives the synthetic data and every
  model; `qp demo` is byte-reproducible.
- **Paper first.** Nothing can reach a real broker unless you implement the
  Alpaca stub *and* set `execution.broker: alpaca`. Don't, until you have months
  of paper results.

## Adding your own signal

Subclass `qp.signals.base.Signal`, implement `generate(close) -> DataFrame` of
scores, register it in `qp/signals/__init__.py:make_signal`, point a config at
it. That's the whole extension surface.

## Using real data

`stooq` and `yfinance` adapters exist but are best-effort (stooq bot-blocks some
networks; yfinance needs the `data` extra). The robust path is `source: csv` —
drop `<SYMBOL>.csv` files (`date,open,high,low,close,volume`) in `data/csv/` and
set `data.csv_dir`.

## Full documentation

`docs/main.pdf` (LaTeX, `docs/` source, build with `cd docs && make pdf`) covers
the architecture, every formula (vol targeting, cost model, metrics), the
synthetic data model, the walk-forward methodology, the demo results with the
tearsheet embedded, the MCP layer, and — most importantly — Chapter 10, the
honest limitations section on why this is career capital and not an income
strategy.

## Connecting to your energy-markets work

The natural next signal is a **power-market model**: ENTSO-E day-ahead prices as
the data source, features from load/renewables/fuel forecasts, target = the
day-ahead vs intraday spread or a storage-dispatch P&L. That reuses the PyPSA /
OR-Tools background and is the exact artefact a trading desk wants to see.
