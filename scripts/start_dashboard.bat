@echo off
title SIH 26077 - Severe Weather Nowcaster Dashboard
color 0b
echo ==============================================================================
echo SIH Problem Statement 26077: Severe Weather Nowcasting (2-6h)
echo Launching Interactive Dashboard locally on http://localhost:8501
echo ==============================================================================
echo.

set SCRIPT_DIR=%~dp0..
cd /d "%SCRIPT_DIR%"

echo [1/2] Checking Python environment...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo [2/2] Starting Streamlit Interactive Dashboard...
echo Dashboard will open automatically in your default browser.
echo.
streamlit run app/main.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause
