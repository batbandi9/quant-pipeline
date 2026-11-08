"""Technical features. Every function takes a wide close panel (index=date,
columns=symbol) and returns a same-shaped DataFrame. All are causal: value at
row ``t`` uses only data up to and including ``t``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(close: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return np.log(close).diff(n)


def realized_vol(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return log_return(close).rolling(n).std() * np.sqrt(252)


def momentum(close: pd.DataFrame, n: int = 60, skip: int = 0) -> pd.DataFrame:
    """Trailing return over ``n`` days, optionally skipping the most recent ``skip``."""
    return close.shift(skip) / close.shift(n + skip) - 1.0


def zscore(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return (close - ma) / sd


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    return line - line.ewm(span=signal, adjust=False).mean()  # histogram


def bollinger_pctb(close: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper, lower = ma + k * sd, ma - k * sd
    return (close - lower) / (upper - lower)


def rolling_skew(close: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    return log_return(close).rolling(n).skew()


# name -> (callable, default kwargs). The token "vol_20" resolves to
# realized_vol(n=20); "mom_60" -> momentum(n=60); etc.
_REGISTRY = {
    "ret": (log_return, {"n": 1}),
    "vol": (realized_vol, {"n": 20}),
    "mom": (momentum, {"n": 60}),
    "zscore": (zscore, {"n": 20}),
    "rsi": (rsi, {"n": 14}),
    "macd": (macd, {}),
    "bbp": (bollinger_pctb, {"n": 20}),
    "skew": (rolling_skew, {"n": 60}),
}


def compute(token: str, close: pd.DataFrame) -> pd.DataFrame:
    """Resolve a feature token like ``"mom_60"`` or ``"rsi_14"`` to a panel."""
    parts = token.split("_")
    base = parts[0]
    if base not in _REGISTRY:
        raise KeyError(f"unknown feature {token!r}; known bases: {sorted(_REGISTRY)}")
    fn, kwargs = _REGISTRY[base]
    kwargs = dict(kwargs)
    if len(parts) > 1 and parts[1].isdigit():
        # the single numeric suffix maps to the function's primary window arg
        primary = "n" if "n" in kwargs or base != "macd" else "fast"
        kwargs[primary] = int(parts[1])
    return fn(close, **{k: v for k, v in kwargs.items() if k in fn.__code__.co_varnames})
