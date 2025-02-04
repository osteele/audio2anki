default:
    @just --list

# Run all checks (linting, type checking, and tests)
check: lint test

# Format code
format:
    uv run --dev ruff format .

fix: format
    uv run --dev ruff check --fix --unsafe-fixes .

# Run linting
lint:
    uv run --dev ruff check .
    uv run --dev pyright .

# Run tests
test *ARGS:
    uv run --dev pytest tests/ {{ARGS}}

# Install the package in development mode
install:
    uv pip install -e ".[dev]"
