@echo off

echo Starting Explainable Credit Risk Assessment...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Please create it with:
    echo python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate

python -m streamlit run app\app.py

pause