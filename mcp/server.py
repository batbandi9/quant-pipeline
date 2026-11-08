"""MCP server exposing the qp pipeline as tools an agent can call.

Install + run
------------
    pip install -e ".[mcp]"          # adds the `mcp` SDK
    python mcp/server.py             # stdio transport

Register in Claude Code (`.mcp.json` in your project, or `claude mcp add`):

    {
      "mcpServers": {
        "qp": { "command": "python", "args": ["mcp/server.py"], "cwd": "/abs/path/to/quant-pipeline" }
      }
    }

Then an agent can: list configs and runs, launch backtests, read metrics and the
text report, check paper-broker status, and step the live loop -- all read-mostly,
with paper execution the only side effect.

This file degrades gracefully: without the `mcp` package it prints setup help and
exits 0, so it never breaks CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "The `mcp` package is not installed.\n"
        "  pip install -e \".[mcp]\"\n"
        "then re-run:  python mcp/server.py",
        file=sys.stderr,
    )
    raise SystemExit(0)

from qp.config import RunConfig  # noqa: E402
from qp.live.loop import run_once  # noqa: E402
from qp.pipeline import run_pipeline  # noqa: E402

mcp = FastMCP("qp", instructions="Systematic trading research pipeline: data -> features -> "
              "signals -> backtest -> risk -> paper execution.")


@mcp.tool()
def describe_pipeline() -> str:
    """Return the pipeline architecture and the available config knobs."""
    return (ROOT / "README.md").read_text()


@mcp.tool()
def list_configs() -> list[str]:
    """List runnable YAML configs under configs/."""
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob("*.yaml"))


@mcp.tool()
def list_runs() -> list[dict]:
    """List completed runs under runs/ with their headline metrics."""
    out = []
    for d in sorted((ROOT / "runs").glob("*/")):
        s = d / "summary.json"
        if s.exists():
            m = json.loads(s.read_text())
            out.append({k: m[k] for k in ("name", "sharpe", "cagr", "max_drawdown", "n_days")
                        if k in m})
    return out


@mcp.tool()
def run_backtest(config_path: str) -> dict:
    """Run a full backtest from a config path (e.g. 'configs/demo.yaml'). Returns metrics."""
    cfg = RunConfig.from_yaml(ROOT / config_path)
    out = run_pipeline(cfg)
    return {"run_dir": out.run_dir, **out.result.metrics}


@mcp.tool()
def get_run_report(name: str) -> str:
    """Return the plain-text metrics report for a completed run."""
    p = ROOT / "runs" / name / "report.txt"
    if not p.exists():
        return f"no report for run {name!r} -- run a backtest first"
    return p.read_text()


@mcp.tool()
def get_tearsheet_path(name: str) -> str:
    """Return the absolute path to a run's tearsheet PNG."""
    p = ROOT / "runs" / name / "tearsheet.png"
    return str(p) if p.exists() else f"no tearsheet for run {name!r}"


@mcp.tool()
def paper_iteration(config_path: str) -> dict:
    """Step the live loop once against the paper broker; returns fills + broker snapshot."""
    cfg = RunConfig.from_yaml(ROOT / config_path)
    return run_once(cfg)


if __name__ == "__main__":
    mcp.run()
