#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "Repository: $REPO_DIR"

mkdir -p data

# If model missing and MODEL_URL provided, try to fetch it before starting
MODEL_PATH="$REPO_DIR/data/streamlit_model.joblib"
if [ ! -f "$MODEL_PATH" ]; then
  if [ -n "${MODEL_URL-}" ]; then
    echo "Model not found locally; downloading from MODEL_URL..."
    # prefer python fetch helper if available
    if [ -f "$REPO_DIR/scripts/fetch_model.py" ]; then
      python3 "$REPO_DIR/scripts/fetch_model.py" --url "$MODEL_URL" --out "$MODEL_PATH" || true
    else
      # fallback to curl
      curl -L "$MODEL_URL" -o "$MODEL_PATH" || true
    fi
    if [ -f "$MODEL_PATH" ]; then
      echo "Model downloaded to $MODEL_PATH"
    else
      echo "Model could not be downloaded. Continuing without model (app will show an error)."
    fi
  else
    echo "Model not present at $MODEL_PATH and MODEL_URL not set. App will run but predictions will fail until model is provided."
  fi
fi

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
