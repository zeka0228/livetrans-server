@echo off
REM Build the Windows audio capture helper (tools\wincap) used by mode B loopback capture.
REM Requires .NET SDK 8+ : https://dotnet.microsoft.com/download
REM Messages are ASCII only (cmd reads .bat in the OEM code page).
setlocal
cd /d "%~dp0.."
where dotnet >nul 2>nul
if errorlevel 1 (
  echo [wincap] dotnet not found. Mode B ^(PC audio capture^) will be unavailable.
  exit /b 1
)
dotnet publish tools\wincap -c Release -o tools\wincap\out --nologo -v q
if errorlevel 1 (
  echo [wincap] build failed.
  exit /b 1
)
echo [wincap] built tools\wincap\out\wincap.exe
endlocal
