@echo off
title QuickDraw AI

echo.
echo  ================================
echo   QuickDraw AI wird gestartet...
echo  ================================
echo.

:: FastAPI Backend in neuem PowerShell-Fenster
echo  [1/2] Starte FastAPI Backend...
start "FastAPI Backend" powershell -NoExit -Command "cd '%~dp0'; .\venv\Scripts\Activate.ps1; uvicorn api:app --port 8000"

:: 4 Sekunden warten
timeout /t 4 /nobreak >nul

:: React Frontend in neuem PowerShell-Fenster
echo  [2/2] Starte React Frontend...
start "React Frontend" powershell -NoExit -Command "cd '%~dp0\quickdraw-ui'; $env:CI='false'; npm start"

echo.
echo  ================================
echo   FastAPI: http://localhost:8000
echo   React:   http://localhost:3000
echo  ================================
echo.
echo  Beide Fenster schliessen zum Beenden.
pause
