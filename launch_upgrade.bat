@echo off
cd /d "%~dp0"
echo Starting Model Upgrade
echo =========================

echo.
echo This will upgrade your model with advanced features and algorithms.
echo.
pause

python -m src.model_upgrade_pipeline

pause
