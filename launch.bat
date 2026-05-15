@echo off
cd /d "%~dp0"
echo ========================================
echo  Clarity in Credit - Loan Risk Assessment
echo ========================================
echo.
echo Choose an option:
echo 1. Start Main Application
echo 2. View Available Launch Options
echo 3. Run Model Upgrade
echo 4. View Documentation
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting main application...
    streamlit run app/app_real_data.py
) else if "%choice%"=="2" (
    echo.
    echo Available launch options in scripts/ folder:
    echo - start_app.bat (Main launcher)
    echo - launch_simple.bat (Port 8501)
    echo - launch_fixed_final.bat (Port 8502)
    echo - launch_fixed_interface.bat (Port 8503)
    echo - launch_neural_network_app.bat (Port 8504)
    echo - quick_fix_launch.bat (Port 8505)
    echo - launch_upgrade.bat (Model upgrade)
    echo.
    pause
) else if "%choice%"=="3" (
    echo Running model upgrade...
    python -m src.model_upgrade_pipeline
    pause
) else if "%choice%"=="4" (
    echo Opening documentation...
    start docs/README.md
) else (
    echo Invalid choice. Please run the script again.
    pause
)
