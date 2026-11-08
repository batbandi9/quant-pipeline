import json
from pathlib import Path

from qp.config import RunConfig
from qp.execution.paper import PaperBroker
from qp.live.loop import run_once
from qp.pipeline import run_pipeline


def _demo_cfg(tmp_path: Path, **over):
    base = {
        "name": "pt",
        "seed": 3,
        "data": {"source": "synthetic",
                 "symbols": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
                 "start": "2016-01-01", "end": "2021-12-31"},
        "signal": {"type": "cross_sectional_momentum", "params": {"lookback": 40, "skip": 3}},
        "outdir": str(tmp_path / "runs"),
        "execution": {"broker": "paper", "starting_equity": 100_000,
                      "state_dir": str(tmp_path / "state")},
    }
    base.update(over)
    return RunConfig.model_validate(base)


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _demo_cfg(tmp_path)
    out = run_pipeline(cfg)
    rd = Path(out.run_dir)
    assert (rd / "tearsheet.png").exists()
    assert (rd / "metrics.json").exists()
    assert (rd / "report.txt").exists()
    m = json.loads((rd / "metrics.json").read_text())
    assert m["n_days"] > 500
    assert -1.0 < m["max_drawdown"] <= 0.0


def test_paper_broker_rebalance_and_resume(tmp_path):
    b = PaperBroker("x", starting_equity=100_000, state_dir=str(tmp_path))
    prices = {"A": 10.0, "B": 20.0}
    b.rebalance_to({"A": 0.5, "B": -0.3}, prices)
    eq1 = b.equity()
    assert 90_000 < eq1 <= 100_000  # only cost drag
    assert b.positions()["A"] > 0 and b.positions()["B"] < 0

    # a fresh broker with the same name resumes state
    b2 = PaperBroker("x", starting_equity=999, state_dir=str(tmp_path))
    assert abs(b2.equity() - eq1) < 1e-6


def test_live_run_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _demo_cfg(tmp_path)
    rec = run_once(cfg)
    assert rec["n_fills"] >= 1
    assert rec["broker"]["equity"] > 0
    assert (tmp_path / "state" / "pt.loop.jsonl").exists()
