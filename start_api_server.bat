@echo off
REM Start the Drug Repurposing Assistant API Server
REM Keep the window open to monitor the server

cd /d "%~dp0"

echo.
echo ===============================================
echo Drug Repurposing Assistant API Server
echo ===============================================
echo.
echo Starting server...
echo API will be available at: http://localhost:8000
echo Interactive docs at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python src/api.py

pause
