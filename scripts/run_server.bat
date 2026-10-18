@echo off
REM Starts the livetrans server (host/port from config/default.yaml + default.local.yaml).
setlocal
cd /d "%~dp0.."
REM Force UTF-8 for stdout/stderr so Korean log lines are not garbled when redirected (ERR-001).
set PYTHONUTF8=1
uv run python -m livetrans
endlocal
