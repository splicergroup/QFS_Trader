#!/bin/bash
# One-click launcher for the Intraday Analysis app.
# Double-click in Finder (macOS) to activate the venv and start Streamlit.
# The script runs from its own folder, so it works no matter where it's launched.

cd "$(dirname "$0")" || exit 1

echo "Activating virtual environment…"
source .venv/bin/activate

echo "Starting Intraday Analysis (press Ctrl+C in this window to stop)…"
streamlit run intraday_analysis_app.py
