# PowerShell script untuk deploy ke Vercel
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "      Preparing Digit Classifier App for Vercel" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Running build script to prepare clean deployment folder..." -ForegroundColor Yellow
python build_app.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error running build script! Aborting." -ForegroundColor Red
    Read-Host -Prompt "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Step 2: Navigating to deploy folder..." -ForegroundColor Yellow
Set-Location -Path deploy

Write-Host ""
Write-Host "Step 3: Initializing local Git repository..." -ForegroundColor Yellow
git init
git add .
git commit -m "Deploy to Vercel"

Write-Host ""
Write-Host "Step 4: Deploying to Vercel..." -ForegroundColor Yellow
vercel --prod

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "Deployment process completed!" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green

Read-Host -Prompt "Press Enter to exit"
