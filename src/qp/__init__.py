"""qp -- a full systematic-trading research pipeline.

Stages
------
data      ingestion + storage (synthetic / CSV / yfinance / stooq)
features  technical + calendar feature engineering
signals   rule-based, ML (sklearn), and sequence (LSTM) alpha models
validation walk-forward and purged K-fold splits
backtest  vectorised multi-asset engine with realistic costs
risk      exposure limits and a drawdown kill-switch
execution paper broker (default) and a real-broker adapter stub
report    tearsheet (PNG) + machine-readable metrics
live      the pull -> features -> signal -> target -> reconcile -> orders loop

An MCP server (see ``mcp/``) is layered on top so an agent can drive the pipeline.
"""

__version__ = "0.1.0"
