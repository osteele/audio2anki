default:
    @just --list

# Run all checks (linting, type checking, and tests)
check: lint typecheck test

# Format code
format:
    uv run --dev ruff format .

fix: format
    uv run --dev ruff check --fix --unsafe-fixes .

# Run linting
lint:
    uv run --dev ruff check .

# Run tests
test *ARGS:
    uv run --dev python -m pytest tests/ {{ARGS}}

# Run type checking
typecheck:
    uv run --dev pyright .

# Install the package in development mode
install:
    uv pip install -e ".[dev]"

run *ARGS:
    uv run audio2anki {{ARGS}}
