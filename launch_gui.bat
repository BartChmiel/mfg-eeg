@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m scripts.gui_app
) else (
    python -m scripts.gui_app
)
