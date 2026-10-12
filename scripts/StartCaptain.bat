@echo off
setlocal
set "PROJECT_DIR=%~dp0.."
if defined NEVC_PYTHON_EXE (
  set "PYTHON_EXE=%NEVC_PYTHON_EXE%"
) else (
  set "PYTHON_EXE=E:\Sim-One\Tools\python36\python.exe"
)
set "LAUNCH_LOG=%PROJECT_DIR%\runtime_data\launcher.log"
set "CONSOLE_LOG=%PROJECT_DIR%\runtime_data\captain-console-%RANDOM%-%RANDOM%.log"
rem Decision API uses m/s: 30 km/h = 8.333333333333334 m/s.
if not defined NEVC_DECISION_CRUISE_SPEED set "NEVC_DECISION_CRUISE_SPEED=8.333333333333334"
rem User-provided SimOne-Car preset: wheelbase 2.9187 m, front overhang 1 m,
rem body width 1.8 m. Competition GPS coordinates refer to the rear axle.
rem Shared decision/planning geometry; preserve caller overrides for other cars.
if not defined NEVC_VEHICLE_FRONT_OFFSET_M set "NEVC_VEHICLE_FRONT_OFFSET_M=3.9187"
if not defined NEVC_VEHICLE_HALF_WIDTH_M set "NEVC_VEHICLE_HALF_WIDTH_M=0.9"
rem The same installed SimOne-Car preset has a rear overhang of 0.88 m.
if not defined NEVC_VEHICLE_REAR_OFFSET_M set "NEVC_VEHICLE_REAR_OFFSET_M=0.88"

if not exist "%PROJECT_DIR%\runtime_data" mkdir "%PROJECT_DIR%\runtime_data"
echo [%date% %time%] StartCaptain invoked: %~f0 >> "%LAUNCH_LOG%"
echo [%date% %time%] Python console log: %CONSOLE_LOG% >> "%LAUNCH_LOG%"
echo [%date% %time%] Decision cruise speed: %NEVC_DECISION_CRUISE_SPEED% m/s >> "%LAUNCH_LOG%"
echo [%date% %time%] Vehicle geometry: front=%NEVC_VEHICLE_FRONT_OFFSET_M% m half-width=%NEVC_VEHICLE_HALF_WIDTH_M% m >> "%LAUNCH_LOG%"
echo [%date% %time%] Vehicle rear offset: %NEVC_VEHICLE_REAR_OFFSET_M% m >> "%LAUNCH_LOG%"
echo [INFO] Captain log: %PROJECT_DIR%\runtime_data\captain.log
echo [INFO] Python console log: %CONSOLE_LOG%
echo [INFO] Decision cruise speed: %NEVC_DECISION_CRUISE_SPEED% m/s
echo [INFO] Vehicle geometry: front=%NEVC_VEHICLE_FRONT_OFFSET_M% m half-width=%NEVC_VEHICLE_HALF_WIDTH_M% m

if not exist "%PYTHON_EXE%" (
  echo [ERROR] SimOne Python not found: %PYTHON_EXE%
  echo [%date% %time%] SimOne Python not found: %PYTHON_EXE% >> "%LAUNCH_LOG%"
  exit /b 1
)

title NEVC Captain Runtime
cd /d "%PROJECT_DIR%"
if exist "%PROJECT_DIR%\config\local.ini" (
  "%PYTHON_EXE%" -u "%PROJECT_DIR%\main.py" --config "%PROJECT_DIR%\config\local.ini" %* >> "%CONSOLE_LOG%" 2>&1
) else (
  "%PYTHON_EXE%" -u "%PROJECT_DIR%\main.py" %* >> "%CONSOLE_LOG%" 2>&1
)
set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] StartCaptain exited with code %EXIT_CODE% >> "%LAUNCH_LOG%"
exit /b %EXIT_CODE%
