"""Sequence alpha model: an LSTM regressor over windowed per-symbol features.

Optional -- requires ``pip install 'qp[ml]'`` (torch). Kept deliberately small
and honest: on real market data a model like this is dominated by transaction
costs and regime shift far more often than it "predicts the market". Use it to
*learn the architecture*, benchmark it against :class:`~qp.signals.rules.Momentum`,
and never trust an in-sample equity curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import build_panels
from ..features.pipeline import cross_sectional_zscore
from ..logging import get_logger
from .base import Signal

log = get_logger(__name__)


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError(
            "sequence signals need torch. Install with `pip install 'qp[ml]'` "
            "or choose signal.type: momentum / cross_sectional_momentum / ml"
        ) from e
    import torch

    return torch


class SequenceSignal(Signal):
    market_neutral = True

    def __init__(
        self,
        features: list[str],
        window: int = 32,
        horizon: int = 5,
        hidden: int = 32,
        epochs: int = 8,
        lr: float = 1e-3,
        train_days: int = 756,
        test_days: int = 126,
        step_days: int = 126,
        embargo_days: int = 5,
        seed: int = 7,
    ):
        self.features = features
        self.window = window
        self.horizon = horizon
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.embargo_days = embargo_days
        self.seed = seed

    # ------------------------------------------------------------------ helpers
    def _tensor_panel(self, close: pd.DataFrame):
        panels = build_panels(close, self.features)
        # shape (T, N_symbols, F)
        arr = np.stack([p.values for p in panels.values()], axis=-1)
        fwd = (close.shift(-self.horizon) / close - 1.0).values
        return arr.astype("float32"), fwd.astype("float32")

    def _windows(self, arr, fwd, idx_range):
        X, Y, meta = [], [], []
        lo, hi = idx_range
        for t in range(max(self.window, lo), hi):
            block = arr[t - self.window : t]  # (window, N, F)
            if np.isnan(block).any():
                continue
            for j in range(block.shape[1]):
                target = fwd[t, j]
                if np.isnan(target):
                    continue
                X.append(block[:, j, :])
                Y.append(target)
                meta.append((t, j))
        if not X:
            return None
        return np.asarray(X), np.asarray(Y), meta

    # ------------------------------------------------------------------ main
    def generate(self, close: pd.DataFrame) -> pd.DataFrame:
        torch = _require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        arr, fwd = self._tensor_panel(close)
        T = arr.shape[0]
        n_feat = arr.shape[-1]
        dates = close.index

        class LSTMReg(torch.nn.Module):
            def __init__(self, n_in, hidden):
                super().__init__()
                self.lstm = torch.nn.LSTM(n_in, hidden, batch_first=True)
                self.head = torch.nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        pred = np.full((T, arr.shape[1]), np.nan, dtype="float32")

        # walk forward over integer index ranges
        i = self.train_days
        while i + self.test_days <= T:
            tr = self._windows(arr, fwd, (self.window, i - self.embargo_days))
            te = self._windows(arr, fwd, (i, i + self.test_days))
            i += self.step_days
            if tr is None or te is None:
                continue
            Xtr, Ytr, _ = tr
            Xte, _, meta_te = te

            model = LSTMReg(n_feat, self.hidden)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = torch.nn.MSELoss()
            xb = torch.tensor(Xtr)
            yb = torch.tensor(Ytr)
            model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                out = model(torch.tensor(Xte)).numpy()
            for (t, j), v in zip(meta_te, out):
                pred[t, j] = v

        wide = pd.DataFrame(pred, index=dates, columns=close.columns)
        z = cross_sectional_zscore(wide)
        gross = z.abs().sum(axis=1).replace(0, np.nan)
        return z.div(gross, axis=0).fillna(0.0)
