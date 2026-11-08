import numpy as np
import pandas as pd
import pytest

from qp.data.synthetic import SyntheticSource
from qp.data.base import to_wide


@pytest.fixture(scope="session")
def long_df():
    return SyntheticSource(seed=7).fetch(
        ["AAA", "BBB", "CCC", "DDD", "EEE"], "2015-01-01", "2022-12-31"
    )


@pytest.fixture(scope="session")
def close(long_df):
    return to_wide(long_df, "close")


@pytest.fixture
def trend_prices():
    """Deterministic: two assets, one up-trending, one down-trending."""
    idx = pd.bdate_range("2018-01-01", periods=500)
    up = pd.Series(100 * (1.0004) ** np.arange(500), index=idx)
    down = pd.Series(100 * (0.9997) ** np.arange(500), index=idx)
    return pd.DataFrame({"UP": up, "DOWN": down})
