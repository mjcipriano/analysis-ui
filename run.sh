#!/usr/bin/env bash
# Set up (first run) and start Analysis UI from a source checkout.
#   ./run.sh            start on http://localhost:8765
#   ./run.sh --port 9000 --no-browser
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x backend/.venv/bin/python ]; then
  echo "› Creating the Python environment (first run only)…"
  python3 -m venv backend/.venv
  backend/.venv/bin/pip install --quiet --upgrade pip
  backend/.venv/bin/pip install --quiet -e "backend[dev]"
fi

if [ ! -f backend/analysis_ui/static/index.html ]; then
  echo "› Building the web app (first run only)…"
  (cd frontend && npm install --no-audit --no-fund --silent && npm run build --silent)
fi

if ! command -v Rscript >/dev/null 2>&1; then
  echo "! R was not found. Python analyses and matplotlib graphs work; install R to enable"
  echo "  ANOVA / mixed models / regression models and ggplot2 graphs (see README: “Installing R”)."
fi

exec backend/.venv/bin/analysis-ui "$@"
