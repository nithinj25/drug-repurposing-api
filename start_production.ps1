# Production Start Script - Drug Repurposing API
# This script starts the API server with production settings

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DRUG REPURPOSING API - PRODUCTION STARTUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check if required packages are installed
Write-Host ""
Write-Host "[2/5] Checking dependencies..." -ForegroundColor Yellow
$requiredPackages = @("fastapi", "uvicorn", "requests")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    $installed = pip show $package 2>$null
    if (-not $installed) {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "  ⚠ Missing packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
    Write-Host "  Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  ✓ All dependencies installed" -ForegroundColor Green
}

# Check for .env file
Write-Host ""
Write-Host "[3/5] Checking environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  ✓ .env file found" -ForegroundColor Green
} else {
    Write-Host "  ⚠ No .env file found" -ForegroundColor Yellow
    Write-Host "    Using default configuration" -ForegroundColor Gray
}

# Kill existing process on port 8000
Write-Host ""
Write-Host "[4/5] Checking port availability..." -ForegroundColor Yellow
$processOnPort = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($processOnPort) {
    Write-Host "  ⚠ Port 8000 is in use. Stopping existing process..." -ForegroundColor Yellow
    $processOnPort | Select-Object -ExpandProperty OwningProcess | ForEach-Object {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
    Write-Host "  ✓ Port 8000 freed" -ForegroundColor Green
} else {
    Write-Host "  ✓ Port 8000 available" -ForegroundColor Green
}

# Get local IP address
$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" } | Select-Object -First 1).IPAddress

# Start the API server
Write-Host ""
Write-Host "[5/5] Starting API server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  API SERVER RUNNING" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Local Access:" -ForegroundColor White
Write-Host "    API:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "    Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Network Access:" -ForegroundColor White
Write-Host "    API:  http://${localIP}:8000" -ForegroundColor Cyan
Write-Host "    Docs: http://${localIP}:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Available Endpoints:" -ForegroundColor White
Write-Host "    POST   /analyze              - Analyze drug repurposing" -ForegroundColor Gray
Write-Host "    POST   /batch                - Batch analysis" -ForegroundColor Gray
Write-Host "    GET    /jobs                 - List all jobs" -ForegroundColor Gray
Write-Host "    GET    /jobs/{job_id}        - Get job status" -ForegroundColor Gray
Write-Host "    GET    /agents               - List available agents" -ForegroundColor Gray
Write-Host "    GET    /health               - Health check" -ForegroundColor Gray
Write-Host ""
Write-Host "  Frontend Configuration:" -ForegroundColor White
Write-Host "    const API_BASE_URL = 'http://localhost:8000';" -ForegroundColor Gray
Write-Host "    // Or for network: 'http://${localIP}:8000'" -ForegroundColor Gray
Write-Host ""
Write-Host "  Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Start the server
python src/api.py
