"""In-memory paper broker with JSON-persisted state.

Fills at the provided last price plus a symmetric slippage in bps. State
(``cash``, ``positions``, ``last_prices``, ``fills``) is written to
``<state_dir>/<name>.json`` after every rebalance so the live loop is resumable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..logging import get_logger
from .base import Broker, Fill

log = get_logger(__name__)


class PaperBroker(Broker):
    def __init__(
        self,
        name: str = "paper",
        starting_equity: float = 100_000.0,
        slippage_bps: float = 3.0,
        commission_bps: float = 1.0,
        state_dir: str = "execution_state",
    ):
        self.name = name
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.state_path = Path(state_dir) / f"{name}.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        if self.state_path.exists():
            s = json.loads(self.state_path.read_text())
            self.cash = s["cash"]
            self.pos = {k: float(v) for k, v in s["positions"].items()}
            self.last = {k: float(v) for k, v in s["last_prices"].items()}
            self.fills = s.get("fills", [])
            log.info("resumed paper broker from %s (equity ~ %.0f)", self.state_path, self.equity())
        else:
            self.cash = float(starting_equity)
            self.pos: dict[str, float] = {}
            self.last: dict[str, float] = {}
            self.fills: list[dict] = []

    # ---------------------------------------------------------------- Broker API
    def equity(self) -> float:
        mv = sum(self.pos.get(s, 0.0) * self.last.get(s, 0.0) for s in self.pos)
        return self.cash + mv

    def positions(self) -> dict[str, float]:
        return dict(self.pos)

    def mark(self, prices: dict[str, float]) -> None:
        self.last.update({k: float(v) for k, v in prices.items()})

    def rebalance_to(
        self, target_weights: dict[str, float], prices: dict[str, float]
    ) -> list[Fill]:
        self.mark(prices)
        eq = self.equity()
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fills: list[Fill] = []

        for sym, w in target_weights.items():
            px = float(prices[sym])
            target_qty = (w * eq) / px
            delta = target_qty - self.pos.get(sym, 0.0)
            if abs(delta * px) < 1.0:  # ignore sub-$1 rebalances
                continue
            slip = px * self.slippage_bps / 1e4 * (1 if delta > 0 else -1)
            fill_px = px + slip
            commission = abs(delta * fill_px) * self.commission_bps / 1e4
            self.cash -= delta * fill_px + commission
            self.pos[sym] = self.pos.get(sym, 0.0) + delta
            f = Fill(sym, delta, fill_px, commission, ts)
            fills.append(f)
            self.fills.append(f.__dict__)

        # close any name not in target
        for sym in list(self.pos):
            if sym not in target_weights and abs(self.pos[sym]) > 1e-9:
                px = float(self.last.get(sym, prices.get(sym, 0.0)))
                if px <= 0:
                    continue
                delta = -self.pos[sym]
                self.cash += delta * px
                self.pos[sym] = 0.0
                self.fills.append(Fill(sym, delta, px, 0.0, ts).__dict__)

        self._persist()
        return fills

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "equity": self.equity(),
            "cash": self.cash,
            "positions": {k: v for k, v in self.pos.items() if abs(v) > 1e-9},
            "n_fills": len(self.fills),
        }

    # ---------------------------------------------------------------- internals
    def _persist(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "cash": self.cash,
                    "positions": self.pos,
                    "last_prices": self.last,
                    "fills": self.fills[-500:],
                    "equity": self.equity(),
                },
                indent=2,
            )
        )
