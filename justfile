default:
    @just --list

# Run all checks (type checking and linting)
check:
    uv run --dev mypy src/
    uv run --dev ruff check src/
    uv run --dev black --check src/

# Format code
fmt:
    uv run --dev black src/
    uv run --dev ruff --fix src/

# Run tests
test:
    uv run --dev pytest tests/

# Install the package in development mode
install:
    uv pip install -e ".[dev]"
