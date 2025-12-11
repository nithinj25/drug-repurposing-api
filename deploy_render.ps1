# Quick Deploy Script - Push to GitHub and Deploy to Render
# Run this after you've connected your GitHub repo to Render

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DEPLOY TO RENDER - QUICK SCRIPT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "[1/5] Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "  ✓ Git initialized" -ForegroundColor Green
} else {
    Write-Host "[1/5] Git repository already initialized" -ForegroundColor Green
}

# Create .gitignore if it doesn't exist
Write-Host ""
Write-Host "[2/5] Checking .gitignore..." -ForegroundColor Yellow
if (-not (Test-Path ".gitignore")) {
    @"
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
.venv
*.log
.DS_Store
.vscode/
.idea/
*.db
*.sqlite3
data/
logs/
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Host "  ✓ .gitignore created" -ForegroundColor Green
} else {
    Write-Host "  ✓ .gitignore exists" -ForegroundColor Green
}

# Add and commit
Write-Host ""
Write-Host "[3/5] Committing changes..." -ForegroundColor Yellow
git add .
$commitMessage = Read-Host "Enter commit message (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = "Deploy to Render - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
}
git commit -m "$commitMessage"
Write-Host "  ✓ Changes committed" -ForegroundColor Green

# Check for remote
Write-Host ""
Write-Host "[4/5] Checking GitHub remote..." -ForegroundColor Yellow
$hasRemote = git remote -v | Select-String "origin"
if (-not $hasRemote) {
    Write-Host "  ⚠ No GitHub remote found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Please create a new repository on GitHub, then run:" -ForegroundColor White
    Write-Host "  git remote add origin https://github.com/YOUR_USERNAME/drug-repurposing-api.git" -ForegroundColor Cyan
    Write-Host "  git push -u origin main" -ForegroundColor Cyan
    Write-Host ""
    exit 0
} else {
    Write-Host "  ✓ GitHub remote configured" -ForegroundColor Green
}

# Push to GitHub
Write-Host ""
Write-Host "[5/5] Pushing to GitHub..." -ForegroundColor Yellow
git push
Write-Host "  ✓ Code pushed to GitHub" -ForegroundColor Green

# Success message
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✓ CODE PUSHED TO GITHUB" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor White
Write-Host ""
Write-Host "1. Go to https://render.com" -ForegroundColor Cyan
Write-Host "2. Click 'New +' → 'Web Service'" -ForegroundColor Cyan
Write-Host "3. Connect your GitHub repository" -ForegroundColor Cyan
Write-Host "4. Configure:" -ForegroundColor Cyan
Write-Host "   - Build Command: pip install -r requirements.txt" -ForegroundColor Gray
Write-Host "   - Start Command: python src/api.py" -ForegroundColor Gray
Write-Host "   - Add Environment Variables:" -ForegroundColor Gray
Write-Host "     • GROQ_API_KEY" -ForegroundColor Gray
Write-Host "     • PUBMED_API_KEY" -ForegroundColor Gray
Write-Host "5. Click 'Create Web Service'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your API will be live in 3-5 minutes! 🚀" -ForegroundColor Green
Write-Host ""
Write-Host "See RENDER_DEPLOYMENT.md for detailed instructions" -ForegroundColor Yellow
Write-Host ""
