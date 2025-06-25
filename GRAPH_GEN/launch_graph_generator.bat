@echo off
REM Bitcoin Graph Generator Launcher
echo Bitcoin Graph Generator
echo =========================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import matplotlib, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

REM Launch the graph generator
echo Launching Bitcoin Graph Generator...
echo.
python btc_graph_generator.py --interactive

pause
