"""Time-series cross-validation that does not leak the future.

* ``walk_forward`` -- expanding-window-free rolling train/test with an embargo
  gap between train end and test start (covers the label horizon).
* ``purged_kfold`` -- Lopez de Prado style: K contiguous test folds, training
  rows within ``embargo`` of any test fold are dropped.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def walk_forward(
    dates: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    step_days: int,
    embargo_days: int = 0,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    n = len(dates)
    start = 0
    while start + train_days + embargo_days + test_days <= n:
        tr0 = dates[start]
        tr1 = dates[start + train_days - 1]
        te0 = dates[start + train_days + embargo_days]
        te1 = dates[min(start + train_days + embargo_days + test_days - 1, n - 1)]
        yield tr0, tr1, te0, te1
        start += step_days


def purged_kfold(
    dates: pd.DatetimeIndex, n_splits: int = 5, embargo_days: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    n = len(dates)
    folds = np.array_split(np.arange(n), n_splits)
    for test_idx in folds:
        t0, t1 = test_idx[0], test_idx[-1]
        lo = max(0, t0 - embargo_days)
        hi = min(n, t1 + embargo_days + 1)
        mask = np.ones(n, dtype=bool)
        mask[lo:hi] = False
        train_idx = np.arange(n)[mask]
        yield train_idx, test_idx
