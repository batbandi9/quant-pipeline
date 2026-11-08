"""``qp`` command-line interface."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import RunConfig
from .logging import get_logger

app = typer.Typer(add_completion=False, help="qp -- systematic trading pipeline")
console = Console()
log = get_logger("qp.cli")

DEMO_CONFIG = {
    "name": "demo",
    "seed": 7,
    "data": {
        "source": "synthetic",
        "symbols": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"],
        "start": "2013-01-01",
        "end": "2024-12-31",
    },
    "features": ["ret_1", "vol_20", "mom_60", "mom_120", "rsi_14", "zscore_20"],
    "signal": {"type": "cross_sectional_momentum", "params": {"lookback": 60, "skip": 5}},
    "portfolio": {"sizing": "vol_target", "target_vol": 0.10, "max_weight": 0.30,
                  "gross_leverage": 1.0, "long_only": False, "rebalance": "weekly"},
    "costs": {"commission_bps": 1.0, "half_spread_bps": 2.0, "borrow_bps_annual": 50.0},
    "risk": {"max_drawdown_stop": 0.25, "max_gross": 1.5, "max_position": 0.35},
}


def _metrics_table(metrics: dict, title: str) -> Table:
    t = Table(title=title, show_header=True, header_style="bold cyan")
    t.add_column("metric")
    t.add_column("value", justify="right")
    order = ["n_days", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
             "calmar", "hit_rate", "avg_daily_turnover", "time_in_market",
             "var_95", "cvar_95"]
    for k in order:
        if k not in metrics:
            continue
        v = metrics[k]
        if k in ("cagr", "ann_vol", "max_drawdown", "hit_rate", "time_in_market",
                 "var_95", "cvar_95"):
            s = f"{v*100:.2f}%"
        elif k == "n_days":
            s = f"{int(v)}"
        else:
            s = f"{v:.2f}"
        t.add_row(k, s)
    return t


@app.command()
def demo():
    """Run the full pipeline end-to-end on synthetic data -> runs/demo/."""
    from .pipeline import run_pipeline

    cfg = RunConfig.model_validate(DEMO_CONFIG)
    out = run_pipeline(cfg)
    console.print(_metrics_table(out.result.metrics, "demo backtest"))
    console.print(f"[green]artefacts:[/green] {out.run_dir}/  (tearsheet.png, report.txt, metrics.json)")


@app.command()
def backtest(config: Path = typer.Option(..., "--config", "-c", exists=True)):
    """Run a backtest from a YAML config."""
    from .pipeline import run_pipeline

    cfg = RunConfig.from_yaml(config)
    out = run_pipeline(cfg)
    console.print(_metrics_table(out.result.metrics, f"{cfg.name} backtest"))
    console.print(f"[green]artefacts:[/green] {out.run_dir}/")


@app.command("data")
def data_cmd(config: Path = typer.Option(..., "--config", "-c", exists=True)):
    """Fetch/generate the configured dataset into data/ and show a preview."""
    from .data import load_prices, to_wide

    cfg = RunConfig.from_yaml(config)
    df = load_prices(cfg.data)
    close = to_wide(df, "close")
    console.print(f"[green]{len(df)}[/green] rows, {close.shape[1]} symbols, "
                  f"{close.index[0].date()} .. {close.index[-1].date()}")
    console.print(close.tail().round(2).to_string())


@app.command()
def walkforward(config: Path = typer.Option(..., "--config", "-c", exists=True)):
    """Compare rule-based vs ML vs (optional) sequence signals on one dataset."""
    from .backtest import run_backtest
    from .data import load_prices, to_wide
    from .signals import make_signal

    cfg = RunConfig.from_yaml(config)
    close = to_wide(load_prices(cfg.data), "close")

    variants = {
        "cs_momentum": {"type": "cross_sectional_momentum", "params": {"lookback": 60, "skip": 5}},
        "ml_gbr": {"type": "ml", "params": {"horizon": 5}},
    }
    if cfg.signal.type == "sequence":
        variants["lstm"] = {"type": "sequence", "params": cfg.signal.params}

    rows = {}
    for name, sig in variants.items():
        c = cfg.model_copy(deep=True)
        c.name = f"{cfg.name}__{name}"
        c.signal = c.signal.model_validate(sig)
        try:
            scores = make_signal(c.signal, c.features, c.validation).generate(close)
            res = run_backtest(close, scores, c)
            rows[name] = res.metrics
        except Exception as e:  # keep going if an optional dep is missing
            log.warning("variant %s failed: %s", name, e)

    t = Table(title=f"walk-forward comparison: {cfg.name}", header_style="bold cyan")
    t.add_column("variant")
    for col in ("sharpe", "cagr", "max_drawdown", "avg_daily_turnover"):
        t.add_column(col, justify="right")
    for name, m in rows.items():
        t.add_row(name, f"{m['sharpe']:.2f}", f"{m['cagr']*100:.1f}%",
                  f"{m['max_drawdown']*100:.1f}%", f"{m.get('avg_daily_turnover', float('nan')):.2f}")
    console.print(t)
    (cfg.run_dir() / "walkforward.json").write_text(json.dumps(rows, indent=2))


@app.command()
def paper(config: Path = typer.Option(..., "--config", "-c", exists=True),
          loop: bool = typer.Option(False, help="run forever instead of one iteration"),
          every: int = typer.Option(86_400, help="seconds between iterations when --loop")):
    """One iteration (or a loop) of the live pipeline against the paper broker."""
    from .live import run_forever, run_once

    cfg = RunConfig.from_yaml(config)
    if loop:
        run_forever(cfg, every_seconds=every)
    else:
        rec = run_once(cfg)
        console.print_json(json.dumps(rec["broker"]))


@app.command()
def report(run: str = typer.Option(..., help="run name under runs/")):
    """Reprint the metrics table for a completed run."""
    p = Path("runs") / run / "metrics.json"
    if not p.exists():
        raise typer.BadParameter(f"no metrics at {p}")
    console.print(_metrics_table(json.loads(p.read_text()), run))


if __name__ == "__main__":
    app()
