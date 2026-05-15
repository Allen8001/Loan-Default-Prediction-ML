@echo off
cd /d "%~dp0"
echo Starting Fixed Final App
echo ========================
streamlit run app/app_real_data.py --server.port=8502
pause