@echo off
TITLE SIH 26077 - AI Severe Weather Nowcasting System
color 0A

echo ==============================================================================
echo  SIH 26077: AI Severe Weather Nowcasting System (2-6 Hours Lead Time)
echo  Launching Local Services: FastAPI (Port 8000) ^& Streamlit (Port 8501)
echo ==============================================================================

:: Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

:: Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment 'venv'...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment '.venv'...
    call .venv\Scripts\activate.bat
)

:: Start FastAPI Backend in background window
echo [INFO] Starting FastAPI Alert Engine on http://localhost:8000 ...
start "SIH 26077 - FastAPI Backend (Port 8000)" cmd /k "uvicorn api.main:app --reload --port 8000"

:: Wait 2 seconds for API initialization
timeout /t 2 /nobreak >nul

:: Start Streamlit Dashboard in primary window
echo [INFO] Starting Streamlit Interactive Dashboard on http://localhost:8501 ...
streamlit run app/main.py --server.port 8501

pause
