"""Broker contract used by the live loop.

The loop thinks in **target weights**; the broker translates them into orders
against its current positions and marks the book to market.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class Fill:
    symbol: str
    qty: float          # signed; + buy, - sell
    price: float
    cost: float         # commission + modelled slippage
    ts: str


class Broker(abc.ABC):
    @abc.abstractmethod
    def equity(self) -> float:
        ...

    @abc.abstractmethod
    def positions(self) -> dict[str, float]:
        """symbol -> share quantity (signed)."""

    @abc.abstractmethod
    def mark(self, prices: dict[str, float]) -> None:
        """Update last prices used for valuation."""

    @abc.abstractmethod
    def rebalance_to(self, target_weights: dict[str, float], prices: dict[str, float]) -> list[Fill]:
        """Trade so post-trade weights match ``target_weights`` (of current equity)."""

    @abc.abstractmethod
    def snapshot(self) -> dict:
        ...
