@echo off
echo ============================================================
echo  Combined Eye Disease Detection API — FastAPI Server
echo ============================================================
echo.
echo Starting Combined Backend on http://localhost:8000
echo Docs available at http://localhost:8000/docs
echo.
cd /d "%~dp0"
call venv\Scripts\activate
cd eye_api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
