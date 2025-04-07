# Development Guide for audio2anki

## Setup

1. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
2. [Install `just`](https://just.systems/man/en/pre-built-binaries.html)
3. Clone the repository and set up the development environment:

```bash
# Clone the repository
git clone https://github.com/osteele/audio2anki.git
cd audio2anki

# Install dependencies
just setup

# Install pre-commit hooks
uv run --dev pre-commit install
```

## Development Commands

```bash
# Run tests
just test

# Format code
just format

# Run type checking
just typecheck

# Run linting
just lint

# Fix linting issues automatically
just fix

# Run all checks (lint, typecheck, test)
just check
```

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run code formatting before each commit.
The hooks will automatically run formatting tools to ensure code quality.

To install the pre-commit hooks:

```bash
uv run --dev pre-commit install
```

After installation, the hooks will run automatically on each commit.

## Publishing

To publish a new version to PyPI:

```bash
just publish
```

This will clean build artifacts, build the package, and publish it to PyPI.

## CI/CD

The project uses GitHub Actions for continuous integration. The workflow runs:
- Linting with ruff
- Type checking with pyright
- Tests with pytest

The workflow runs on multiple Python versions (3.10, 3.11, 3.12) to ensure compatibility.
