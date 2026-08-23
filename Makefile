.PHONY: sync test lint format format-check typecheck check

sync:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff check --fix .
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run pyright

check: lint format-check typecheck test