@echo off
REM ============================================================================
REM  push_to_github.bat  -  double-clickable launcher for push_to_github.ps1
REM ============================================================================
REM  Why this exists
REM  ---------------
REM  Double-clicking a .ps1 in Windows Explorer does NOT run it - by default
REM  Windows opens it in Notepad. And even from a terminal, PowerShell's
REM  execution policy normally blocks unsigned local scripts. This wrapper
REM  solves both problems:
REM
REM    * it invokes PowerShell explicitly with -ExecutionPolicy Bypass, which
REM      applies to THIS PROCESS ONLY and never changes the machine's policy;
REM    * it pauses at the end, so the window stays open long enough to read
REM      the result. Without that, a double-clicked script flashes shut
REM      instantly - including when it fails, which is exactly when you most
REM      need to see the output.
REM
REM  What it does
REM  ------------
REM  Always runs a DRY RUN first (changes nothing, uploads nothing), shows you
REM  the result, and only then asks whether to publish for real. There is no
REM  way to accidentally publish by double-clicking.
REM
REM  For anything beyond the defaults (-Private, -Tag, -RepoName, -Branch ...)
REM  run the PowerShell script directly instead:
REM
REM      pwsh -ExecutionPolicy Bypass -File push_to_github.ps1 -Tag v4.0
REM
REM  Requirements: this .bat must sit in the SAME folder as push_to_github.ps1.
REM ============================================================================

setlocal

REM %~dp0 is this file's own folder, with a trailing backslash. Using it means
REM the script works regardless of the current directory when launched - which
REM for a double-click is often C:\Windows\System32, not the project folder.
set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%push_to_github.ps1"

echo.
echo =========================================================
echo   JasMAP4  -  Publish to GitHub
echo =========================================================
echo.

if not exist "%PS_SCRIPT%" (
    echo ERROR: Could not find push_to_github.ps1
    echo.
    echo   Looked here: %PS_SCRIPT%
    echo.
    echo   This .bat must be in the SAME folder as push_to_github.ps1.
    echo.
    goto finish
)

REM Prefer PowerShell 7+ (pwsh) when installed; otherwise fall back to the
REM Windows built-in Windows PowerShell 5.1. Both run this script correctly.
set "PS_EXE=powershell"
where pwsh >nul 2>&1
if %ERRORLEVEL%==0 set "PS_EXE=pwsh"
echo Using: %PS_EXE%

echo.
echo ---------------------------------------------------------
echo  STEP 1 of 2 - DRY RUN (nothing is changed or uploaded)
echo ---------------------------------------------------------
echo.

"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -DryRun

echo.
echo ---------------------------------------------------------
echo  Dry run finished. Nothing has been published yet.
echo  Read the output above before continuing.
echo ---------------------------------------------------------
echo.

set "CONFIRM=n"
set /p CONFIRM="Publish for real now? (y/N): "

if /I "%CONFIRM%"=="y"   goto publish
if /I "%CONFIRM%"=="yes" goto publish

echo.
echo Aborted - nothing was published.
goto finish

:publish
echo.
echo ---------------------------------------------------------
echo  STEP 2 of 2 - PUBLISHING
echo ---------------------------------------------------------
echo.
echo  If prompted for a password, paste a GitHub Personal Access
echo  Token - NOT your account password. Create one at:
echo      https://github.com/settings/tokens    (scope: repo)
echo.

"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%"

if not "%ERRORLEVEL%"=="0" (
    echo.
    echo The script exited with code %ERRORLEVEL% - see the messages above.
)

:finish
echo.
echo Press any key to close this window...
pause >nul
endlocal
