"""Signal registry + factory keyed by ``SignalCfg``."""
from __future__ import annotations

from ..config import SignalCfg, ValidationCfg
from .base import Signal
from .ml import MLSignal
from .rules import CrossSectionalMomentum, MeanReversion, Momentum
from .sequence import SequenceSignal

__all__ = [
    "Signal",
    "Momentum",
    "MeanReversion",
    "CrossSectionalMomentum",
    "MLSignal",
    "SequenceSignal",
    "make_signal",
]


def make_signal(cfg: SignalCfg, features: list[str], validation: ValidationCfg) -> Signal:
    p = dict(cfg.params)
    if cfg.type == "momentum":
        return Momentum(**p)
    if cfg.type == "mean_reversion":
        return MeanReversion(**p)
    if cfg.type == "cross_sectional_momentum":
        return CrossSectionalMomentum(**p)
    if cfg.type == "ml":
        return MLSignal(
            features=features,
            train_days=validation.train_days,
            test_days=validation.test_days,
            step_days=validation.step_days,
            embargo_days=validation.embargo_days,
            **p,
        )
    if cfg.type == "sequence":
        return SequenceSignal(
            features=features,
            train_days=validation.train_days,
            test_days=validation.test_days,
            step_days=validation.step_days,
            embargo_days=validation.embargo_days,
            **p,
        )
    raise ValueError(f"unknown signal type {cfg.type!r}")
