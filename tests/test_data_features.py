import pandas as pd

from qp.data.synthetic import SyntheticSource
from qp.features import technical as tech
from qp.features.pipeline import build_ml_dataset


def test_synthetic_is_deterministic():
    a = SyntheticSource(seed=42).fetch(["X", "Y"], "2019-01-01", "2021-01-01")
    b = SyntheticSource(seed=42).fetch(["X", "Y"], "2019-01-01", "2021-01-01")
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_schema_and_positivity(long_df):
    assert list(long_df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert (long_df[["open", "high", "low", "close"]] > 0).all().all()
    assert (long_df["high"] >= long_df[["open", "close"]].max(axis=1) - 1e-6).all()
    assert (long_df["low"] <= long_df[["open", "close"]].min(axis=1) + 1e-6).all()


def test_rsi_bounds(close):
    r = tech.rsi(close, 14).dropna()
    assert r.min().min() >= 0 and r.max().max() <= 100


def test_zscore_is_centered(close):
    z = tech.zscore(close, 20).dropna()
    assert abs(z.mean().mean()) < 0.3


def test_feature_token_resolution(close):
    assert tech.compute("mom_60", close).shape == close.shape
    assert tech.compute("rsi_14", close).shape == close.shape


def test_ml_dataset_no_leakage(close):
    X, y = build_ml_dataset(close, ["mom_20", "vol_20", "rsi_14"], horizon=5)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert X.index.equals(y.index)
    # target is a *forward* return -> last `horizon` dates cannot have labels
    last_date = close.index[-1]
    assert last_date not in y.index.get_level_values("date")
