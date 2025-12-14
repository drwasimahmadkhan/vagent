@echo off
echo Starting CSV Analysis Frontend...
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Frontend starting on http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app/frontend/app.py
