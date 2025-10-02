# Cross-platform Makefile for Windows (cmd/PowerShell) and Unix shells
# Uses the venv interpreter directly to avoid activation differences

VENV=.venv
PY=$(VENV)/Scripts/python.exe
PIP=$(VENV)/Scripts/pip.exe
PRECOMMIT=$(VENV)/Scripts/pre-commit.exe
PYTEST=$(VENV)/Scripts/python.exe -m pytest
RUFF=$(VENV)/Scripts/python.exe -m ruff
BLACK=$(VENV)/Scripts/python.exe -m black
ISORT=$(VENV)/Scripts/python.exe -m isort
MYPY=$(VENV)/Scripts/python.exe -m mypy
LINT_IMPORTS=$(VENV)/Scripts/lint-imports.exe

.PHONY: init fmt lint types imports test qa

init:
	python -m venv $(VENV)
	$(PIP) install -e ".[dev]"
	$(PRECOMMIT) install -t pre-commit -t pre-push

fmt:
	$(BLACK) .
	$(ISORT) .
	$(RUFF) format .

lint:
	$(RUFF) .

types:
	$(MYPY)

imports:
	$(LINT_IMPORTS) --config importlinter.ini

test:
	$(PYTEST) -q

qa: fmt lint types imports test
