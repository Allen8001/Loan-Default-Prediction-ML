@echo off
cd /d "%~dp0"
echo Install, train, and start Clarity in Credit (Ctrl+C to stop the app)
python run.py all %*
pause
