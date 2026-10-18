@echo off
REM livetrans-server setup: creates .venv and installs dependencies with uv.
REM Messages are ASCII only (cmd reads .bat in the OEM code page).
setlocal
cd /d "%~dp0.."
where uv >nul 2>nul
if errorlevel 1 (
  echo [setup] uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/
  exit /b 1
)
uv sync
if errorlevel 1 (
  echo [setup] uv sync failed.
  exit /b 1
)
call scripts\build_wincap.bat
if errorlevel 1 echo [setup] wincap skipped: mode B capture unavailable, other features work.
echo [setup] done. Run scripts\run_server.bat
endlocal
