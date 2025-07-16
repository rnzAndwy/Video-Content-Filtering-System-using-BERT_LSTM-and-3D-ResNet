@echo off
title Tensorr1 Development Environment
echo ========================================
echo  Starting Tensorr1 Development Server
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "final_environment.yml" (
    echo ERROR: final_environment.yml not found!
    echo Make sure you're in the tensorr-deployment folder.
    echo.
    pause
    exit /b 1
)

REM Check if conda is available
conda --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda not found!
    echo Please install Anaconda or Miniconda first.
    echo Download: https://www.anaconda.com/products/distribution
    echo.
    pause
    exit /b 1
)

REM Initialize conda for batch script usage
echo Initializing conda...
call conda init cmd.exe
if errorlevel 1 (
    echo ERROR: Failed to initialize conda!
    echo.
    pause
    exit /b 1
)

REM Refresh environment variables after conda init
echo Refreshing environment...
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" 2>nul || call "%USERPROFILE%\miniconda3\Scripts\activate.bat" 2>nul

REM Check if environment exists, create if it doesn't
conda info --envs | findstr "tensorr1" >nul 2>&1
if errorlevel 1 (
    echo Environment 'tensorr1' not found. Creating it now...
    echo This may take a few minutes...
    echo.
    conda env create -f final_environment.yml
    if errorlevel 1 (
        echo ERROR: Failed to create environment!
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Environment created successfully!
    echo.
)

REM Activate environment
echo Activating tensorr1 environment...
call conda activate tensorr1
if errorlevel 1 (
    echo ERROR: Failed to activate environment!
    echo Try running this script as Administrator or manually run: conda init cmd.exe
    echo.
    pause
    exit /b 1
)

REM Verify key packages
echo Verifying installation...
python -c "import torch; import tensorflow; import cv2; print('✓ All key packages loaded successfully')" 2>nul
if errorlevel 1 (
    echo WARNING: Some packages may not be properly installed.
    echo The application may still work, but please check for errors.
    echo.
)

REM Start the application
echo.
echo ========================================
echo  Starting Your Application...
echo ========================================
echo.

REM Replace 'main.py' with your actual main file
python main.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ========================================
    echo  Application stopped with errors
    echo ========================================
    echo.
    pause
) else (
    echo.
    echo ========================================
    echo  Application finished successfully
    echo ========================================
    echo.
)