"""Typed configuration for a pipeline run.

A run is fully described by one YAML file. Load with ``RunConfig.from_yaml(path)``.
Every stage of the pipeline reads only its own sub-model, so configs stay flat
and reviewable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SyntheticCfg(BaseModel):
    seed: int = 7
    mu: float = 0.06           # annual drift
    sigma: float = 0.20        # annual vol
    regime_switch: bool = True  # inject bull/bear/chop regimes
    corr: float = 0.3          # average pairwise correlation


class DataCfg(BaseModel):
    source: Literal["synthetic", "csv", "yfinance", "stooq"] = "synthetic"
    symbols: list[str] = Field(default_factory=lambda: ["AAA", "BBB", "CCC", "DDD", "EEE"])
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    csv_dir: str = "data/csv"                 # for source == csv
    synthetic: SyntheticCfg = SyntheticCfg()
    cache: bool = True


class SignalCfg(BaseModel):
    type: Literal[
        "momentum", "mean_reversion", "cross_sectional_momentum", "ml", "sequence"
    ] = "cross_sectional_momentum"
    params: dict[str, Any] = Field(default_factory=dict)


class PortfolioCfg(BaseModel):
    sizing: Literal["fixed", "vol_target", "risk_parity"] = "vol_target"
    target_vol: float = 0.10          # annualised, for vol_target / risk_parity
    max_weight: float = 0.25          # per-name cap after sizing
    gross_leverage: float = 1.0       # sum(|w|) target
    long_only: bool = False
    rebalance: Literal["daily", "weekly", "monthly"] = "daily"
    vol_lookback: int = 60            # window for ex-ante portfolio-vol estimate
    max_leverage_scale: float = 3.0   # cap on the vol-target multiplier


class CostsCfg(BaseModel):
    commission_bps: float = 1.0       # per unit turnover
    half_spread_bps: float = 2.0      # per unit turnover
    borrow_bps_annual: float = 50.0   # on short exposure
    slippage_bps: float = 0.0         # extra, applied like spread


class RiskCfg(BaseModel):
    max_drawdown_stop: float = 0.35   # flatten the book past this equity drawdown
    max_gross: float = 1.5
    max_position: float = 0.30
    vol_lookback: int = 20


class ValidationCfg(BaseModel):
    scheme: Literal["none", "walk_forward", "purged_kfold"] = "walk_forward"
    train_days: int = 756
    test_days: int = 63
    step_days: int = 63
    embargo_days: int = 5
    n_splits: int = 5                 # for purged_kfold


class ExecutionCfg(BaseModel):
    broker: Literal["paper", "alpaca"] = "paper"
    starting_equity: float = 100_000.0
    state_dir: str = "execution_state"


class RunConfig(BaseModel):
    name: str = "run"
    seed: int = 7
    data: DataCfg = DataCfg()
    features: list[str] = Field(
        default_factory=lambda: ["ret_1", "vol_20", "mom_60", "rsi_14", "zscore_20"]
    )
    signal: SignalCfg = SignalCfg()
    portfolio: PortfolioCfg = PortfolioCfg()
    costs: CostsCfg = CostsCfg()
    risk: RiskCfg = RiskCfg()
    validation: ValidationCfg = ValidationCfg()
    execution: ExecutionCfg = ExecutionCfg()
    outdir: str = "runs"

    @model_validator(mode="after")
    def _propagate_seed(self) -> "RunConfig":
        # a single knob controls determinism everywhere
        self.data.synthetic.seed = self.seed
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(raw)

    def run_dir(self) -> Path:
        d = Path(self.outdir) / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
