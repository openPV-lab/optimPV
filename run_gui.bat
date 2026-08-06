@echo off
REM Launches the optimPV process-correlation / condition-search GUI in your browser.
cd /d "%~dp0"
python -m streamlit run gui\app.py
