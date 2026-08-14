@echo off
REM LLAMADX run script - AMD DX12 backend for llama.cpp
REM
REM Usage:
REM   run.bat "C:\path\to\model.gguf" [extra llama-cli args]
REM   run.bat "C:\path\to\model.gguf" -ncmoe 20        (MoE model, offload experts to CPU)
REM   run.bat "C:\path\to\model.gguf" -ngl 20           (partial GPU offload)
REM
REM Or hardcode a default model path below and just run "run.bat" with no args.

setlocal enabledelayedexpansion

set DEFAULT_MODEL=

set MODEL=%~1
if "%MODEL%"=="" set MODEL=%DEFAULT_MODEL%

if "%MODEL%"=="" (
    echo Usage: run.bat "path\to\model.gguf" [extra args]
    echo   or edit DEFAULT_MODEL= in this file to set a default.
    echo.
    echo Available devices:
    llama-cli.exe --list-devices
    exit /b 1
)

if not exist "%MODEL%" (
    echo Model file not found: %MODEL%
    exit /b 1
)

REM Extra args passed after the model path (e.g. -ncmoe 20, -ngl 99, -c 4096)
set EXTRA=%2 %3 %4 %5 %6 %7 %8 %9

echo Model:  %MODEL%
echo Device: DX120 (AMD RX 9070 XT / 6700 XT tested)
echo.

llama-cli.exe -m "%MODEL%" -dev DX120 -ngl 99 -st %EXTRA%

endlocal
