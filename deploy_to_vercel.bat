@echo off
echo =======================================================
echo      Preparing Digit Classifier App for Vercel
echo =======================================================
echo.

echo Step 1: Running build script to prepare clean deployment folder...
python build_app.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running build script! Aborting.
    pause
    exit /b 1
)

echo.
echo Step 2: Navigating to deploy folder...
cd deploy

echo.
echo Step 3: Initializing local Git repository...
git init
git add .
git commit -m "Deploy to Vercel"

echo.
echo Step 4: Deploying to Vercel...
vercel --prod

echo.
echo =======================================================
echo Deployment process completed!
echo =======================================================

pause
