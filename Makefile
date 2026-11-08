# qp -- systematic trading pipeline

PY ?= python3

.PHONY: help install install-all demo backtest walkforward paper test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## editable install, core deps only
	$(PY) -m pip install -e .

install-all: ## editable install with ml, data, opt, mcp, dev extras
	$(PY) -m pip install -e ".[ml,data,opt,mcp,dev]"

demo: ## end-to-end run on synthetic data -> runs/demo/
	$(PY) -m qp.cli demo

backtest: ## run a backtest from a config: make backtest CONFIG=configs/demo.yaml
	$(PY) -m qp.cli backtest --config $(CONFIG)

walkforward: ## walk-forward ML backtest: make walkforward CONFIG=configs/ml_walkforward.yaml
	$(PY) -m qp.cli walkforward --config $(CONFIG)

paper: ## one iteration of the live loop against the paper broker
	$(PY) -m qp.cli paper --config $(CONFIG)

test: ## run the test suite (synthetic data, no network)
	$(PY) -m pytest

lint: ## ruff check
	$(PY) -m ruff check src tests

clean: ## remove run artefacts and caches
	rm -rf runs/ data/ execution_state/ .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
