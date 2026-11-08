"""Project logging: one rich console handler, module-level ``get_logger``."""
from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def configure(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)
