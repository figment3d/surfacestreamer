@echo off

REM ---- Start Vite in new window ----
start "Vite Dev" cmd /k "npm run dev"

REM ---- Give Vite time to spin up ----
timeout /t 2 >nul

REM ---- Launch Edge to viewer ----
start msedge http://localhost:5173

REM ---- Launch Fake MCU UDP source ----
start "Fake MCU UDP" cmd /k "python fake_mcu_udp.py"

REM ---- Run Python server here ----
if "%1"=="" (
    python ws_bezier_server.py
) else (
    python %1
)