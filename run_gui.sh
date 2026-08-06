#!/usr/bin/env bash
# Launches the optimPV process-correlation / condition-search GUI in your browser.
# Usage: ./run_gui.sh
set -e
cd "$(dirname "$0")"
python3 -m streamlit run gui/app.py
