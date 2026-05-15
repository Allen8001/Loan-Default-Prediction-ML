# Clarity in Credit — minimal Makefile for Unix / GitHub Actions
# Usage: make install | make train | make app | make all
# Override Python: make PYTHON=python3.11 train

PYTHON ?= python3
PORT ?= 8505
# Set TRAIN_SAMPLE=N for a faster training run (e.g. make train TRAIN_SAMPLE=30000)
TRAIN_SAMPLE ?=

.PHONY: install train app all help

help:
	@echo "Targets:"
	@echo "  make install   - pip install -r requirements.txt"
	@echo "  make train     - train XGBoost -> models/loan_default_model_real.pkl"
	@echo "  make app       - streamlit run app/app_real_data.py"
	@echo "  make all       - install, train, then app (blocking)"
	@echo "Optional: TRAIN_SAMPLE=30000 make train"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	@if [ -n "$(TRAIN_SAMPLE)" ]; then \
		$(PYTHON) -m src.train_default_model --sample $(TRAIN_SAMPLE); \
	else \
		$(PYTHON) -m src.train_default_model; \
	fi

app:
	$(PYTHON) -m streamlit run app/app_real_data.py --server.port=$(PORT)

all: install train app
