@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_gtx1050ti.ps1" %*
if errorlevel 1 (
    echo Setup failed. Read the error above and docs\GTX1050TI_SETUP.md.
    pause
    exit /b 1
)
pause
