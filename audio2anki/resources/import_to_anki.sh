#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Add uv to PATH for this session
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# Run the add2anki tool
echo "Importing cards to Anki..."
uvx add2anki@0.1.2 deck.csv --tags audio2anki

echo "Import complete!"
