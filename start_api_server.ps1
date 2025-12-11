# Start Drug Repurposing Assistant API Server
# Run with: powershell -ExecutionPolicy Bypass -File start_api_server.ps1

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Drug Repurposing Assistant API Server" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Starting API server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "API will be available at:" -ForegroundColor Green
Write-Host "  Main API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Swagger UI: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  ReDoc: http://localhost:8000/redoc" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
python src/api.py

Write-Host ""
Write-Host "Server stopped." -ForegroundColor Yellow
