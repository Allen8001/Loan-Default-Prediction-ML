@echo off
cd /d "%~dp0"
echo Quick Fix Launch
echo ================
streamlit run app/app_real_data.py --server.port=8505
pause