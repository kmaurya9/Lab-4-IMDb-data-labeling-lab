# Lab 4 — IMDb Data Labeling: common commands
# Usage: make <target>     (override the interpreter with: make PYTHON=python <target>)

PYTHON ?= python3
NOTEBOOKS = 01_Data_Labeling 02_Data_Augmentation 03_Model_Development

.PHONY: help install test lint run clean

help:
	@echo "make install   install dependencies and NLTK data"
	@echo "make test      run unit tests"
	@echo "make lint      byte-compile the src/ modules"
	@echo "make run       execute all notebooks in order"
	@echo "make clean     remove generated data, outputs and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m nltk.downloader wordnet omw-1.4

test:
	$(PYTHON) -m pytest tests -v

lint:
	$(PYTHON) -m py_compile src/utils.py src/config.py src/data_validator.py

run:
	cd notebooks && for nb in $(NOTEBOOKS); do \
		$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace $$nb.ipynb || exit 1; \
	done

clean:
	rm -f data/imdb_test_gold.csv data/imdb_weak_labeled.csv data/imdb_augmented.csv
	rm -rf outputs/figures outputs/*.pkl outputs/*.csv
	rm -rf .pytest_cache src/__pycache__ tests/__pycache__ notebooks/.ipynb_checkpoints
