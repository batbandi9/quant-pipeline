from ..config import ExecutionCfg
from .base import Broker, Fill
from .paper import PaperBroker

__all__ = ["Broker", "Fill", "PaperBroker", "make_broker"]


def make_broker(cfg: ExecutionCfg, name: str, slippage_bps: float = 3.0) -> Broker:
    if cfg.broker == "paper":
        return PaperBroker(
            name=name,
            starting_equity=cfg.starting_equity,
            slippage_bps=slippage_bps,
            state_dir=cfg.state_dir,
        )
    if cfg.broker == "alpaca":
        from .alpaca import AlpacaBroker

        return AlpacaBroker()
    raise ValueError(f"unknown broker {cfg.broker!r}")
