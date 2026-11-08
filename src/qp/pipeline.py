"""Top-level orchestration: config -> prices -> scores -> backtest -> tearsheet."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import RunConfig
from .data import load_prices, to_wide
from .logging import get_logger
from .report import write_tearsheet
from .signals import make_signal

log = get_logger(__name__)


@dataclass
class PipelineOutput:
    close: pd.DataFrame
    scores: pd.DataFrame
    result: BacktestResult
    run_dir: str


def run_pipeline(cfg: RunConfig, make_report: bool = True) -> PipelineOutput:
    t0 = time.time()
    long_df = load_prices(cfg.data)
    close = to_wide(long_df, "close")
    log.info("prices: %d bars x %d symbols (%s .. %s)",
             len(close), close.shape[1], close.index[0].date(), close.index[-1].date())

    signal = make_signal(cfg.signal, cfg.features, cfg.validation)
    log.info("signal: %r", signal)
    scores = signal.generate(close)

    result = run_backtest(close, scores, cfg)

    run_dir = cfg.run_dir()
    (run_dir / "config.resolved.json").write_text(cfg.model_dump_json(indent=2))
    if make_report:
        png = write_tearsheet(result, run_dir, title=cfg.name)
        log.info("tearsheet -> %s", png)

    (run_dir / "summary.json").write_text(
        json.dumps({"name": cfg.name, "elapsed_s": round(time.time() - t0, 2),
                    **result.metrics}, indent=2)
    )
    return PipelineOutput(close=close, scores=scores, result=result, run_dir=str(run_dir))
