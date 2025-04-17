#!/bin/bash

ADD2ANKI_MIN_VERSION="0.1.2"

# Function to compare versions (returns 0 if $1 >= $2)
version_gte() {
  [ "$1" = "$2" ] && return 0
  local IFS=.
  local i ver1=($1) ver2=($2)
  # Fill empty fields in ver1 with zeros
  for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
    ver1[i]=0
  done
  for ((i=0; i<${#ver1[@]}; i++)); do
    if [[ -z ${ver2[i]} ]]; then
      # Fill empty fields in ver2 with zeros
      ver2[i]=0
    fi
    if ((10#${ver1[i]} > 10#${ver2[i]})); then
      return 0
    fi
    if ((10#${ver1[i]} < 10#${ver2[i]})); then
      return 1
    fi
  done
  return 0
}

# Check if add2anki is installed and meets version requirement
if command -v add2anki &> /dev/null; then
  ADD2ANKI_VERSION=$(add2anki --version 2>/dev/null | head -n1)
  if version_gte "$ADD2ANKI_VERSION" "$ADD2ANKI_MIN_VERSION"; then
    echo "Using system add2anki ($ADD2ANKI_VERSION)"
    add2anki deck.csv --tags audio2anki
    echo "Import complete!"
    exit 0
  else
    echo "Found add2anki ($ADD2ANKI_VERSION), but version is too old. Required: $ADD2ANKI_MIN_VERSION."
    echo "Please upgrade with: pip install -U add2anki"
  fi
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Add uv to PATH for this session
  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Importing cards to Anki via uvx add2anki@0.1.2..."
uvx add2anki@0.1.2 deck.csv --tags audio2anki

echo "Import complete!"
