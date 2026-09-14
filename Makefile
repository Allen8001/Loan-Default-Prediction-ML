PYTHON ?= python

.PHONY: help install train app explain

help:
	@echo "Available commands:"
	@echo "  make install  - Install project dependencies"
	@echo "  make train    - Train and save the credit risk model"
	@echo "  make app      - Start the Streamlit dashboard"
	@echo "  make explain  - Run the SHAP explanation demo"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.train_model

app:
	$(PYTHON) -m streamlit run app/app.py

explain:
	$(PYTHON) -m src.demo_explanation