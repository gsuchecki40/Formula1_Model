#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "Repository: $REPO_DIR"

if [ -f .venv/bin/activate ]; then
  echo "Activating existing virtualenv .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No virtualenv found at .venv — creating and installing requirements"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
fi

echo "Starting Streamlit..."
exec streamlit run app.py
