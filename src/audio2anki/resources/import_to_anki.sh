#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Add uv to PATH for this session
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check if Anki is running
if ! pgrep -x "Anki" > /dev/null; then
  echo "Starting Anki..."
  open -a Anki
  # Wait for Anki to start
  sleep 5
fi

# Run the add2anki tool
echo "Importing cards to Anki..."
uvx git+https://github.com/osteele/add2anki@release deck.csv --tags audio2anki

echo "Import complete!"
