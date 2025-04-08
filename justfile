default:
    @just --list

# Run all checks (linting, type checking, and tests)
check: lint typecheck test

# Clean up build artifacts
clean:
    rm -rf dist

# Format code
format:
    uv run --dev ruff format audio2anki tests

fix: format
    uv run --dev ruff check --fix --unsafe-fixes audio2anki tests

# Run linting
lint:
    uv run --dev ruff check audio2anki tests

# Publish to PyPI
publish: clean
    uv build
    uv publish

run *ARGS:
    uv run audio2anki {{ARGS}}

# Run tests
test *ARGS:
    uv run --dev python -m pytest tests/ {{ARGS}}

# Run type checking
typecheck:
    uv run --dev pyright audio2anki tests
