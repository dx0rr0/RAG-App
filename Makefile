.DEFAULT_GOAL := help

VENV_DIR ?= .venv

ifeq ($(OS),Windows_NT)
PYTHON ?= py
VENV_PYTHON ?= $(VENV_DIR)/Scripts/python.exe
else
PYTHON ?= python3
VENV_PYTHON ?= $(VENV_DIR)/bin/python
endif

.PHONY: help setup serve test benchmark check

help:
	@echo Video RAG local commands:
	@echo   make setup      Create .venv and install the app and test dependencies
	@echo   make serve      Run the local web app and API at http://127.0.0.1:8000
	@echo   make test       Run the offline unittest suite
	@echo   make benchmark  Run the deterministic retrieval benchmark
	@echo   make check      Run tests and benchmark

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install -e ".[test]"

serve:
	$(VENV_PYTHON) -m rag_app.presentation.web

test:
	$(VENV_PYTHON) -B -m unittest discover -v

benchmark:
	$(VENV_PYTHON) -B -m tests.retrieval_benchmark

check: test benchmark
