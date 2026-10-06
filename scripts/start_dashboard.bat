@echo off
title SIH 26077 - Severe Weather Nowcaster Dashboard
color 0b
echo ==============================================================================
echo SIH Problem Statement 26077: Severe Weather Nowcasting (2-6h)
echo High-Speed Interactive Dashboard Launcher (http://localhost:8501)
echo ==============================================================================
echo.

set SCRIPT_DIR=%~dp0..
cd /d "%SCRIPT_DIR%"

rem Optimization flags for instant Python & Streamlit execution
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=0
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
set STREAMLIT_SERVER_RUN_ON_SAVE=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_CLIENT_TOOLBAR_MODE=minimal

echo [1/2] Verifying Python runtime...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo [2/2] Launching Dashboard with Fast-Start Engine...
echo Opening interactive application in default web browser...
echo.

python -m streamlit run app/main.py --server.port 8501 --server.headless false --browser.gatherUsageStats false --server.fileWatcherType none --server.runOnSave false --client.toolbarMode minimal

pause
