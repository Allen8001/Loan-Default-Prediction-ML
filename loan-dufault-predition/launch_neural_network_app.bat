@echo off
cd /d "%~dp0"
echo Starting Neural Network App
echo ============================
streamlit run app/app_real_data.py --server.port=8504
pause