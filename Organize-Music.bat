@echo off
REM ============================================================
REM  Music organizer launcher
REM  - Double-click this file  -> organizes NEW files in C:\Users\chenm\Music
REM    (already-done files are skipped via progress.csv)
REM  - Drag a FOLDER onto this file -> organizes that folder instead
REM  Output always goes to:  C:\Users\chenm\Music\Organized
REM ============================================================

call "C:\Users\chenm\anaconda3\condabin\conda.bat" activate music
if errorlevel 1 (
    echo Could not activate conda env "music".
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Scanning default folder: C:\Users\chenm\Music
    python "C:\Users\chenm\music_organizer\organize.py"
) else (
    echo Scanning dropped folder: %~1
    python "C:\Users\chenm\music_organizer\organize.py" --src "%~1"
)

echo.
echo Done. Organized files are in C:\Users\chenm\Music\Organized
pause
