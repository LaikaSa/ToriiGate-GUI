@echo off
call venv\Scripts\activate
set QUANT_MODE=8bit
python src/main.py
if errorlevel 1 pause