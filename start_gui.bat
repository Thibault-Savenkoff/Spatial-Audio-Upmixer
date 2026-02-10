@echo off
setlocal enabledelayedexpansion

set CONFIG_FILE=%~dp0python312_path.cfg
set PYTHON312=

:: --- Strategy 1: Check if PATH python is 3.12 ---
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do set PYMAJMIN=%%a.%%b

if "%PYMAJMIN%"=="3.12" (
    set PYTHON312=python
    goto :found
)

:: --- Strategy 2: Check saved path from previous run ---
if exist "%CONFIG_FILE%" (
    set /p PYTHON312=<"%CONFIG_FILE%"
    if exist "!PYTHON312!" (
        :: Verify it's actually 3.12
        for /f "tokens=2 delims= " %%v in ('"!PYTHON312!" --version 2^>^&1') do set PYVER=%%v
        for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do set PYMAJMIN=%%a.%%b
        if "!PYMAJMIN!"=="3.12" (
            echo Using saved Python 3.12 path: !PYTHON312!
            goto :found
        )
    )
    echo Saved Python 3.12 path is no longer valid. Searching again...
    del "%CONFIG_FILE%" 2>nul
    set PYTHON312=
)

:: --- Strategy 3: Check common install locations ---
for %%p in (
    "%LocalAppData%\Programs\Python\Python312\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles(x86)%\Python312\python.exe"
    "C:\Python312\python.exe"
    "%LocalAppData%\Programs\Python\Python3.12\python.exe"
) do (
    if exist %%p (
        set PYTHON312=%%~p
        for /f "tokens=2 delims= " %%v in ('"!PYTHON312!" --version 2^>^&1') do set PYVER=%%v
        for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do set PYMAJMIN=%%a.%%b
        if "!PYMAJMIN!"=="3.12" (
            echo Found Python 3.12 at: !PYTHON312!
            echo !PYTHON312!> "%CONFIG_FILE%"
            goto :found
        )
    )
)

:: --- Strategy 4: Ask the user ---
echo ============================================================
echo   Python 3.12 is required (torch 2.5.1 needs it).
echo   Your PATH Python is %PYVER%, which is not compatible.
echo.
echo   Install Python 3.12 from:
echo   https://www.python.org/downloads/release/python-3128/
echo   (Do NOT add to PATH if you want to keep %PYVER% as default)
echo ============================================================
echo.

:ask_path
echo Enter the full path to python.exe for Python 3.12
echo   Example: C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe
echo.
set /p PYTHON312="Path: "

:: Remove surrounding quotes if present
set PYTHON312=!PYTHON312:"=!

if not exist "!PYTHON312!" (
    echo.
    echo   ERROR: File not found: !PYTHON312!
    echo.
    goto :ask_path
)

:: Verify version
for /f "tokens=2 delims= " %%v in ('"!PYTHON312!" --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do set PYMAJMIN=%%a.%%b

if not "!PYMAJMIN!"=="3.12" (
    echo.
    echo   ERROR: That is Python !PYVER!, not 3.12.
    echo.
    goto :ask_path
)

:: Save for next time
echo !PYTHON312!> "%CONFIG_FILE%"
echo.
echo   Path saved to python312_path.cfg for future launches.
echo.

:found
:: Create venv if it doesn't exist
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Creating virtual environment with Python 3.12...
    "!PYTHON312!" -m venv "%~dp0.venv"
)

echo Starting Spatial Audio Upmixer...
echo Installing / updating dependencies...
"%~dp0.venv\Scripts\pip.exe" install -r "%~dp0requirements.txt" --quiet
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0gui.py"
pause
