# MCP layer

`server.py` wraps the pipeline as an [MCP](https://modelcontextprotocol.io)
server so an agent (Claude Code, Claude Desktop, or any MCP client) can drive it.

## Tools exposed

| Tool | What it does | Side effects |
|---|---|---|
| `describe_pipeline()` | returns this project's README (architecture + config schema) | none |
| `list_configs()` | lists `configs/*.yaml` | none |
| `list_runs()` | lists `runs/*` with Sharpe / CAGR / maxDD | none |
| `run_backtest(config_path)` | full data→features→signal→backtest→tearsheet | writes `runs/<name>/` |
| `get_run_report(name)` | plain-text metrics table for a run | none |
| `get_tearsheet_path(name)` | absolute path to the PNG | none |
| `paper_iteration(config_path)` | one live-loop step against the **paper** broker | writes `execution_state/` |

Nothing here can touch a real broker — `execution.broker` must be `paper`, and
the `alpaca` adapter is an unimplemented stub.

## Run it

```
pip install -e ".[mcp]"
python mcp/server.py            # stdio
```

Register in Claude Code — `.mcp.json` at the project root:

```json
{
  "mcpServers": {
    "qp": {
      "command": "python",
      "args": ["mcp/server.py"],
      "cwd": "/absolute/path/to/quant-pipeline"
    }
  }
}
```

Then `/mcp` in an interactive session shows `qp` connected, and the tools appear
as `mcp__qp__run_backtest`, etc.

## Roadmap for this layer

1. **Now** — read-mostly tools above; agent proposes configs, reads results.
2. **Next** — `propose_signal(description)` that asks the calling agent to draft a
   new `Signal` subclass, drops it in `src/qp/signals/`, and backtests it.
3. **Later** — a `research_report(run_a, run_b)` tool that diffs two runs and
   writes a markdown memo; a `sweep(param_grid)` tool for walk-forward parameter
   studies with a results table.
4. **Integration** — the same tools work behind Google Stitch / any orchestration
   layer; the MCP contract is the stable interface, the pipeline internals can
   change freely.
