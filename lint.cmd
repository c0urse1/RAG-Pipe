@echo off
setlocal

REM Simple Windows wrapper to run Ruff linter via the project venv
IF NOT EXIST ".venv\Scripts\python.exe" (
  echo [lint] Missing virtual environment: .venv\Scripts\python.exe
  echo Create it and install dev deps first:
  echo.  python -m venv .venv
  echo.  .venv\Scripts\pip.exe install -e ".[dev]"
  exit /b 1
)

set PY=.venv\Scripts\python.exe

"%PY%" -m ruff . %*

endlocal
