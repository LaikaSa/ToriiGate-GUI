@echo off
call venv\Scripts\activate
python src/main.py
if errorlevel 1 pause
