@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" static_image_server.py %*
if errorlevel 1 pause
