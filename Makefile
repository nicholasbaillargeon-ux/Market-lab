PY := .venv/bin/python
PIP := uv pip install --python .venv

.PHONY: help install db-up db-down demo test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install the package + deps into the uv venv
	uv venv --python 3.14 || true
	$(PIP) -e ".[dev]"

db-up:  ## Start the Postgres container
	docker compose up -d postgres

db-down:  ## Stop the Postgres container
	docker compose down

demo:  ## End-to-end: ingest sample data, run both engines, print the pitfall report
	$(PY) scripts/demo.py

test:  ## Run the test suite (look-ahead, costs, engine parity, ...)
	$(PY) -m pytest -q

clean:  ## Remove the data lake and caches
	rm -rf data/lake .pytest_cache **/__pycache__
