@echo off
echo ======================================================
echo    Football Player Comparison Tool - Starting up...
echo ======================================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and try again.
    echo.
    pause
    exit /b 1
)

REM Check if needed packages are installed
echo Checking dependencies...
pip show streamlit > nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install required packages.
        echo Please run 'pip install -r requirements.txt' manually.
        echo.
        pause
        exit /b 1
    )
)

echo All dependencies are installed.
echo.
echo Starting Streamlit application...
echo The application will open in your default web browser.
echo.
echo Press Ctrl+C in this window to stop the application.
echo.

streamlit run app.py

echo.
echo Application closed.
pause 