"""The live loop: one iteration = pull -> features -> signal -> target weights
-> risk limits -> reconcile against broker -> orders -> record.

``run_once`` is safe to call from cron. ``run_forever`` is a thin scheduler for
a long-running process. Default broker is paper; nothing here can touch real
money unless the config explicitly says ``execution.broker: alpaca`` and that
stub is implemented.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


from ..backtest.portfolio import build_weights
from ..config import RunConfig
from ..data import load_prices, to_wide
from ..execution import make_broker
from ..logging import get_logger
from ..risk.limits import apply_limits
from ..signals import make_signal

log = get_logger(__name__)


def run_once(cfg: RunConfig) -> dict:
    long_df = load_prices(cfg.data)
    close = to_wide(long_df, "close")

    signal = make_signal(cfg.signal, cfg.features, cfg.validation)
    scores = signal.generate(close)

    target_w = build_weights(scores, close, cfg.portfolio)
    target_w = apply_limits(target_w, cfg.risk)

    last_row = target_w.iloc[-1]
    last_px = close.iloc[-1]
    targets = {s: float(last_row[s]) for s in close.columns if abs(last_row[s]) > 1e-6}
    prices = {s: float(last_px[s]) for s in close.columns}

    broker = make_broker(cfg.execution, name=cfg.name)
    fills = broker.rebalance_to(targets, prices)

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_bar": str(close.index[-1].date()),
        "targets": targets,
        "n_fills": len(fills),
        "fills": [f.__dict__ for f in fills],
        "broker": broker.snapshot(),
    }
    log_dir = Path(cfg.execution.state_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{cfg.name}.loop.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")

    log.info(
        "loop %s | equity %.0f | %d fills | %d target names",
        cfg.name, rec["broker"]["equity"], len(fills), len(targets),
    )
    return rec


def run_forever(cfg: RunConfig, every_seconds: int = 86_400) -> None:  # pragma: no cover
    log.info("live loop for %s every %ds (Ctrl-C to stop)", cfg.name, every_seconds)
    while True:
        try:
            run_once(cfg)
        except Exception:
            log.exception("loop iteration failed; continuing")
        time.sleep(every_seconds)
