@echo off
echo Mario AI - Reinforcement Learning System
echo =====================================
echo.

:: Wechsle zum Projektverzeichnis
cd /d "%~dp0"

:: Überprüfe ob Python verfügbar ist
python --version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Python ist nicht installiert oder nicht im PATH verfügbar.
    echo Bitte installieren Sie Python 3.8+ von https://python.org
    pause
    exit /b 1
)

:: Überprüfe ob Virtual Environment existiert
if not exist ".venv" (
    echo Erstelle Python Virtual Environment...
    python -m venv .venv
    if errorlevel 1 (
        echo FEHLER: Konnte Virtual Environment nicht erstellen.
        pausevvvvvvv
        exit /b 1
    )
)

:: Aktiviere Virtual Environment
echo Aktiviere Virtual Environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo FEHLER: Konnte Virtual Environment nicht aktivieren.
    pause
    exit /b 1
)

:: Installiere/Update Dependencies
echo Installiere/Aktualisiere Dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo FEHLER: Konnte Dependencies nicht installieren.
    pause
    exit /b 1
)

:: Starte die Anwendung
echo.
echo Starte Mario AI System...
echo.
python main.py

:: Deaktiviere Virtual Environment
deactivate

echo.
echo Anwendung beendet.
pause
