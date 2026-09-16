@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" phone_server.py --http --host 127.0.0.1 --port 8080 %*
if errorlevel 1 pause
