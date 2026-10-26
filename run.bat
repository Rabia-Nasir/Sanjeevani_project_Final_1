@echo off
chcp 65001 > nul
title Sanjeevani AI Platform
echo ======================================================================
echo       Sanjeevani AI - Clinical Affective Intelligence Platform
echo ======================================================================
echo.

:: Free ports 8000 and 7860 if occupied by lingering processes
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860 "') do taskkill /f /pid %%a >nul 2>&1

:: Check if dependencies need installation
python -c "import fastapi, uvicorn, gradio, librosa" >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/2] Installing required dependencies from requirements.txt...
    pip install -r requirements.txt
    echo.
) else (
    echo [1/2] Dependencies verified.
    echo.
)

echo [2/2] Starting Sanjeevani AI Platform (FastAPI + Gradio Dashboard)...
echo  - Patient Portal:  http://localhost:8000/portal
echo  - REST API Docs:   http://localhost:8000/docs
echo  - Evaluation UI:   http://localhost:7860
echo.

:: Open the Patient Portal in default browser after 3 seconds
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000/portal"

:: Run platform
set PYTHONIOENCODING=utf-8
python run_sanjeevani_platform.py
pause
