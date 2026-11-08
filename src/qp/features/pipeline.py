"""Assemble feature panels and ML datasets.

* ``build_panels`` -> ``{token: wide DataFrame}`` for signal models that want
  cross-sectional feature panels.
* ``build_ml_dataset`` -> tidy ``(X, y)`` with a ``(date, symbol)`` MultiIndex,
  forward-return target, for sklearn / sequence models. The target is shifted so
  row ``t`` predicts the return realised over ``(t, t + horizon]`` -- no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import technical as tech


def build_panels(close: pd.DataFrame, tokens: list[str]) -> dict[str, pd.DataFrame]:
    return {tok: tech.compute(tok, close) for tok in tokens}


def build_ml_dataset(
    close: pd.DataFrame,
    tokens: list[str],
    horizon: int = 5,
    classification: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    panels = build_panels(close, tokens)

    # stack each feature panel to a (date, symbol) Series, then concat columns
    feat_cols = {
        tok: panel.stack(future_stack=True).rename(tok) for tok, panel in panels.items()
    }
    X = pd.concat(feat_cols.values(), axis=1)

    fwd = close.shift(-horizon) / close - 1.0
    y = fwd.stack(future_stack=True)
    y.name = "target"

    df = X.join(y, how="inner").dropna()
    X_out = df[list(tokens)]
    y_out = df["target"]
    if classification:
        y_out = (y_out > 0).astype(int)
    X_out.index.names = ["date", "symbol"]
    y_out.index.names = ["date", "symbol"]
    return X_out, y_out


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Standardise each row across symbols -> market-neutral raw scores."""
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mu, axis=0).div(sd, axis=0)
