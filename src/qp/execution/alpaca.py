"""Real-broker adapter stub (Alpaca).

Deliberately not implemented. Wiring a live broker is the point where paper
trading has proven the pipeline for months and real money is at stake -- do it
consciously, not by accident. When you are ready:

    pip install alpaca-py
    export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
    # paper endpoint first: https://paper-api.alpaca.markets

then implement the four Broker methods below against ``alpaca.trading.client``.
Keep the paper broker as the default in every config.
"""
from __future__ import annotations

from .base import Broker, Fill


class AlpacaBroker(Broker):
    def __init__(self, *_, **__):
        raise NotImplementedError(
            "AlpacaBroker is a stub. Trade on PaperBroker until you have a "
            "months-long paper track record, then implement this against alpaca-py."
        )

    def equity(self) -> float: ...          # pragma: no cover
    def positions(self) -> dict[str, float]: ...  # pragma: no cover
    def mark(self, prices: dict[str, float]) -> None: ...  # pragma: no cover
    def rebalance_to(self, target_weights, prices) -> list[Fill]: ...  # pragma: no cover
    def snapshot(self) -> dict: ...          # pragma: no cover
