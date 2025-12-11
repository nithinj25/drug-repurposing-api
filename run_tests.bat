@echo off
REM ============================================================================
REM Drug Repurposing Assistant API - Demo Runner
REM This script will run the API examples
REM ============================================================================

echo.
echo ============================================================================
echo   DRUG REPURPOSING ASSISTANT - API DRY RUN
echo ============================================================================
echo.

cd /d "%~dp0"

REM Check if API is running
echo Checking if API server is running...
timeout /t 2 /nobreak > nul

python test_simple.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Could not run tests. Make sure API server is running:
    echo   python src/api.py
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo   TESTS COMPLETED!
echo ============================================================================
echo.
echo Next steps:
echo   1. Run: python src/api.py (if not running)
echo   2. Visit: http://localhost:8000/docs
echo   3. Try more examples: python run_api_examples.py
echo.
pause
