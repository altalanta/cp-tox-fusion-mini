.PHONY: help env download-data download-bbbc021 download-tox21 segment features qc descriptors fuse train error lint tests

PYTHON := python
SRC := src
DATA_DIR := data
REPORTS_DIR := reports

help:
	@echo "cp-tox-fusion-mini commands"
	@echo "  make env                 # Create conda environment"
	@echo "  make download-data       # Download BBBC021 subset and Tox21 CSV"
	@echo "  make segment             # Run Cellpose segmentation"
	@echo "  make features            # Extract morphology features"
	@echo "  make qc                  # Compute QC metrics"
	@echo "  make descriptors         # Build RDKit descriptors"
	@echo "  make fuse                # Align modalities and compute splits"
	@echo "  make train               # Train image/chem/fusion models"
	@echo "  make error               # Run error analysis"
	@echo "  make tests               # Run pytest suite"
	@echo "  make lint                # Run flake8"

env:
	conda env create -f env.yml

download-data: download-bbbc021 download-tox21

download-bbbc021:
	$(PYTHON) -m $(SRC).download_bbbc021 --output $(DATA_DIR)/raw/bbbc021

download-tox21:
	$(PYTHON) -m $(SRC).download_tox21 --output $(DATA_DIR)/raw/tox21

segment:
	$(PYTHON) -m $(SRC).segment_cellpose --input_dir $(DATA_DIR)/raw/bbbc021 --output_dir $(DATA_DIR)/interim/masks

features:
	$(PYTHON) -m $(SRC).extract_cp_features --image_dir $(DATA_DIR)/raw/bbbc021 --mask_dir $(DATA_DIR)/interim/masks --output_dir $(DATA_DIR)/processed

qc:
	$(PYTHON) -m $(SRC).qc_metrics --image_dir $(DATA_DIR)/raw/bbbc021 --mask_dir $(DATA_DIR)/interim/masks --output $(DATA_DIR)/processed/qc_metrics.parquet

descriptors:
	$(PYTHON) -m $(SRC).build_descriptors --input $(DATA_DIR)/raw/tox21/tox21_mini.csv --output $(DATA_DIR)/processed/rdkit_features.parquet

fuse:
	$(PYTHON) -m $(SRC).fuse_modalities --cp_features $(DATA_DIR)/processed/cp_features.parquet --chem_features $(DATA_DIR)/processed/rdkit_features.parquet --output_dir $(DATA_DIR)/processed

train:
	$(PYTHON) -m $(SRC).train_models --processed_dir $(DATA_DIR)/processed --raw_image_dir $(DATA_DIR)/raw/bbbc021 --splits $(DATA_DIR)/processed/splits.json --output_dir $(REPORTS_DIR)

error:
	$(PYTHON) -m $(SRC).error_analysis --processed_dir $(DATA_DIR)/processed --predictions_dir $(DATA_DIR)/processed/predictions --reports_dir $(REPORTS_DIR)

lint:
	flake8 src tests

tests:
	pytest
