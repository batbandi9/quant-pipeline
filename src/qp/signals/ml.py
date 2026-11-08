"""ML alpha model: a scikit-learn regressor over technical features, trained
walk-forward. Predicts forward return; the cross-sectional rank of the
prediction becomes the score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.pipeline import build_ml_dataset, cross_sectional_zscore
from ..logging import get_logger
from ..validation.splits import walk_forward
from .base import Signal

log = get_logger(__name__)


def _default_model() -> Pipeline:
    # HistGBR: fast, handles the low signal-to-noise gracefully with shallow trees
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "hgbr",
                HistGradientBoostingRegressor(
                    max_iter=150,
                    max_depth=3,
                    learning_rate=0.03,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=0,
                ),
            ),
        ]
    )


class MLSignal(Signal):
    market_neutral = True

    def __init__(
        self,
        features: list[str],
        horizon: int = 5,
        train_days: int = 756,
        test_days: int = 63,
        step_days: int = 63,
        embargo_days: int = 5,
        model: Pipeline | None = None,
    ):
        self.features = features
        self.horizon = horizon
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.embargo_days = embargo_days
        self.model_factory = (lambda: model) if model is not None else _default_model
        self.oos_pred_: pd.Series | None = None

    def generate(self, close: pd.DataFrame) -> pd.DataFrame:
        X, y = build_ml_dataset(close, self.features, horizon=self.horizon)
        dates = X.index.get_level_values("date").unique().sort_values()

        preds = []
        for tr_start, tr_end, te_start, te_end in walk_forward(
            dates, self.train_days, self.test_days, self.step_days, self.embargo_days
        ):
            tr = X.loc[(slice(tr_start, tr_end), slice(None)), :]
            te = X.loc[(slice(te_start, te_end), slice(None)), :]
            ytr = y.loc[tr.index]
            if len(tr) < 200 or te.empty:
                continue
            mdl = self.model_factory()
            mdl.fit(tr.values, ytr.values)
            p = pd.Series(mdl.predict(te.values), index=te.index)
            preds.append(p)

        if not preds:
            log.warning("MLSignal produced no OOS predictions; returning flat scores")
            return pd.DataFrame(0.0, index=close.index, columns=close.columns)

        oos = pd.concat(preds).sort_index()
        self.oos_pred_ = oos
        wide = oos.unstack("symbol").reindex(index=close.index, columns=close.columns)
        z = cross_sectional_zscore(wide)
        gross = z.abs().sum(axis=1).replace(0, np.nan)
        return z.div(gross, axis=0).fillna(0.0)
