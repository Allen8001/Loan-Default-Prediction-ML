@echo off
cd /d "%~dp0"
echo LOAN MODEL UPGRADE LAUNCHER
echo ================================

echo.
echo This will upgrade your loan default prediction model with:
echo - Advanced feature engineering (60+ features)
echo - Multiple ML algorithms (XGBoost, LightGBM, CatBoost, Random Forest)
echo - Hyperparameter optimization
echo - Ensemble methods
echo - Cross-validation
echo.
echo Expected improvements:
echo - AUC: 73.7%% → 80-85%%
echo - Features: 28 → 60+
echo - Algorithms: 1 → 5+ with ensemble
echo.

pause

echo.
echo Checking dependencies...
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, catboost; print('All dependencies available')" 2>nul
if errorlevel 1 (
    echo Missing dependencies. Installing...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Running validation tests...
python -m src.test_upgrade
if errorlevel 1 (
    echo Validation tests failed
    pause
    exit /b 1
)

echo.
echo Starting model upgrade...
python -m src.model_upgrade_pipeline

echo.
echo Upgrade process completed.
echo Check the results and generated files.
pause
