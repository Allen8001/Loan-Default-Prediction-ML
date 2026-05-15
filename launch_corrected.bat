@echo off
cd /d "%~dp0"
echo ========================================
echo  Clarity in Credit - CORRECTED VERSION
echo ========================================
echo.
echo Starting corrected application with proper feature relationships...
echo.
echo ✅ Higher Credit Score → Lower Default Risk
echo ✅ Lower DTI → Lower Default Risk  
echo ✅ Proper feature scaling applied
echo.

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check if Streamlit is installed
python -c "import streamlit" >nul 2>nul
if %errorlevel% neq 0 (
    echo Streamlit is not installed. Installing now...
    pip install streamlit
    if %errorlevel% neq 0 (
        echo Error: Failed to install Streamlit.
        pause
        exit /b 1
    )
)

:: Check if requirements are installed
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install project dependencies.
    pause
    exit /b 1
)

echo Starting corrected Streamlit application...
streamlit run app/app_corrected.py --server.port=8506 --server.headless=true

echo.
echo Application should be accessible at http://localhost:8506
echo Press Ctrl+C to stop the application.
pause
