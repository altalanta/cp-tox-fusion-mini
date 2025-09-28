# CP-Tox-Mini Makefile
# Production-ready Cell Painting × Toxicity fusion pipeline

.PHONY: help setup clean data features fuse train eval diagnostics ic50 report all test lint format check docs serve

# Default target
.DEFAULT_GOAL := help

# Python and environment settings
PYTHON := python3
PIP := pip
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

# Data and output directories  
DATA_DIR := data
REPORTS_DIR := reports
DOCS_DIR := docs
MANIFESTS_DIR := manifests

# Pipeline parameters
N_SAMPLES := 100
MODEL_TYPE := logistic
RANDOM_SEED := 42

# Set deterministic environment
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

help: ## Show this help message
	@echo "CP-Tox-Mini Pipeline Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup    # Set up environment and dependencies"
	@echo "  make all      # Run complete pipeline"
	@echo "  make test     # Run tests"
	@echo ""

setup: ## Set up Python environment and install dependencies
	@echo "🔧 Setting up environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e .
	$(VENV_PIP) install -r requirements.txt
	@echo "✅ Environment setup complete"

setup-dev: setup ## Set up development environment with testing tools
	@echo "🔧 Setting up development environment..."
	$(VENV_PIP) install pytest pytest-cov pytest-xdist black flake8 mypy
	@echo "✅ Development environment ready"

clean: ## Clean generated files and cache
	@echo "🧹 Cleaning up..."
	rm -rf $(DATA_DIR)/processed/
	rm -rf $(DATA_DIR)/interim/
	rm -rf $(REPORTS_DIR)/
	rm -rf $(DOCS_DIR)/
	rm -rf .pytest_cache/
	rm -rf cp_tox_mini/__pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

data: ## Download datasets and generate manifest
	@echo "🔽 Downloading data..."
	$(PYTHON) -m cp_tox_mini.cli download --data-dir $(DATA_DIR)
	@echo "✅ Data download complete"

features: data ## Compute Cell Painting and chemical features
	@echo "🧪 Computing features..."
	$(PYTHON) -m cp_tox_mini.cli features --data-dir $(DATA_DIR) --n-samples $(N_SAMPLES)
	@echo "✅ Feature extraction complete"

fuse: features ## Join modalities and create splits
	@echo "🔗 Fusing modalities..."
	$(PYTHON) -m cp_tox_mini.cli fuse
	@echo "✅ Data fusion complete"

train: fuse ## Train baseline model
	@echo "🤖 Training model..."
	$(PYTHON) -m cp_tox_mini.cli train --model-type $(MODEL_TYPE)
	@echo "✅ Model training complete"

eval: train ## Evaluate model and generate metrics
	@echo "📊 Evaluating model..."
	$(PYTHON) -m cp_tox_mini.cli eval-model --output-dir $(REPORTS_DIR) --model-type $(MODEL_TYPE)
	@echo "✅ Model evaluation complete"

diagnostics: fuse ## Run leakage and batch diagnostics
	@echo "🔍 Running diagnostics..."
	$(PYTHON) -m cp_tox_mini.cli diagnostics-cmd --output-path $(REPORTS_DIR)/leakage.json
	@echo "✅ Diagnostics complete"

ic50: ## Run IC50 dose-response analysis
	@echo "💊 Running IC50 analysis..."
	$(PYTHON) -m cp_tox_mini.cli ic50 --output-dir $(REPORTS_DIR)
	@echo "✅ IC50 analysis complete"

report: eval diagnostics ic50 ## Build reports and documentation
	@echo "📝 Building reports..."
	$(PYTHON) -m cp_tox_mini.cli report --reports-dir $(REPORTS_DIR) --docs-dir $(DOCS_DIR)
	@echo "✅ Reports generated"
	@echo "🌐 View at: $(REPORTS_DIR)/index.html"

all: ## Run complete pipeline from data download to report generation
	@echo "🚀 Running complete CP-Tox-Mini pipeline..."
	@echo ""
	$(PYTHON) -m cp_tox_mini.cli all-pipeline --n-samples $(N_SAMPLES)
	@echo ""
	@echo "🎉 Pipeline completed successfully!"
	@echo "📊 View results at: $(REPORTS_DIR)/index.html"
	@echo "🌐 GitHub Pages ready at: $(DOCS_DIR)/index.html"

# Testing targets
test: ## Run test suite
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest tests/ -v
	@echo "✅ Tests completed"

test-cov: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	$(PYTHON) -m pytest tests/ -v --cov=cp_tox_mini --cov-report=term-missing --cov-report=html
	@echo "✅ Tests with coverage completed"

test-fast: ## Run fast tests only (skip slow integration tests)
	@echo "🧪 Running fast tests..."
	$(PYTHON) -m pytest tests/ -v -m "not slow"
	@echo "✅ Fast tests completed"

# Code quality targets
lint: ## Check code style with flake8
	@echo "🔍 Checking code style..."
	$(PYTHON) -m flake8 cp_tox_mini/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "✅ Linting passed"

format: ## Format code with black
	@echo "✨ Formatting code..."
	$(PYTHON) -m black cp_tox_mini/ tests/ --line-length=100
	@echo "✅ Code formatted"

mypy: ## Run type checking with mypy
	@echo "🔍 Running type checks..."
	$(PYTHON) -m mypy cp_tox_mini/ --ignore-missing-imports
	@echo "✅ Type checking passed"

check: lint mypy test ## Run all code quality checks
	@echo "✅ All quality checks passed"

# Documentation targets
docs: report ## Generate documentation (alias for report)

serve-docs: docs ## Serve documentation locally
	@echo "🌐 Serving documentation at http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	cd $(DOCS_DIR) && $(PYTHON) -m http.server 8000

# Status and validation targets
status: ## Show pipeline status and available outputs
	@echo "📊 Pipeline Status:"
	$(PYTHON) -m cp_tox_mini.cli status

validate: ## Validate data integrity against manifest
	@echo "🔍 Validating data..."
	$(PYTHON) -m cp_tox_mini.cli validate
	@echo "✅ Validation complete"

# Utility targets
install: ## Install package in development mode
	@echo "📦 Installing package..."
	$(PIP) install -e .
	@echo "✅ Package installed"

uninstall: ## Uninstall package
	@echo "🗑️  Uninstalling package..."
	$(PIP) uninstall cp-tox-mini -y
	@echo "✅ Package uninstalled"

requirements: ## Generate/update requirements.txt from current environment
	@echo "📋 Generating requirements.txt..."
	$(PIP) freeze > requirements.txt
	@echo "✅ Requirements updated"

# CI/CD targets
ci-test: ## Run tests in CI environment
	@echo "🧪 Running CI tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short --maxfail=1
	@echo "✅ CI tests passed"

ci-build: setup all test ## Complete CI build pipeline
	@echo "🏗️  Running complete CI build..."
	@echo "✅ CI build completed"

# Development helpers
dev-setup: setup-dev ## Set up complete development environment
	@echo "🔧 Development environment ready!"
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

demo: clean all ## Run clean demo from scratch
	@echo "🎭 Running clean demo..."
	@echo "✅ Demo completed - check $(REPORTS_DIR)/index.html"

quick: features eval report ## Quick pipeline for development (skip slow steps)
	@echo "⚡ Quick pipeline completed"

# Debug targets
debug-data: ## Show data summary and diagnostics
	@echo "🔍 Data summary:"
	@if [ -f "$(DATA_DIR)/processed/features.parquet" ]; then \
		$(PYTHON) -c "import pandas as pd; df=pd.read_parquet('$(DATA_DIR)/processed/features.parquet'); print(f'Features: {df.shape}'); print(df.head())"; \
	else \
		echo "No features data found - run 'make features' first"; \
	fi

debug-reports: ## Show reports summary
	@echo "📊 Reports summary:"
	@ls -la $(REPORTS_DIR)/ 2>/dev/null || echo "No reports found - run 'make report' first"

# Performance targets
profile: ## Run pipeline with profiling
	@echo "⏱️  Running with profiler..."
	$(PYTHON) -m cProfile -o profile_stats.prof -m cp_tox_mini.cli all-pipeline --n-samples 50
	@echo "✅ Profiling completed - check profile_stats.prof"

benchmark: ## Run performance benchmark
	@echo "⏱️  Running benchmark..."
	time make quick
	@echo "✅ Benchmark completed"

# Help for specific components
help-cli: ## Show CLI help
	$(PYTHON) -m cp_tox_mini.cli --help

help-data: ## Show data format requirements
	@echo "📋 Data Format Requirements:"
	@echo "  - Features: Parquet files with numeric features + metadata"
	@echo "  - Target: Binary classification (0/1)"
	@echo "  - Metadata: compound_id, plate_id, well_row, well_col"
	@echo "  - Manifest: SHA256 hashes for data integrity"

# Cleanup variants
clean-data: ## Clean only data files
	rm -rf $(DATA_DIR)/processed/ $(DATA_DIR)/interim/

clean-reports: ## Clean only reports
	rm -rf $(REPORTS_DIR)/ $(DOCS_DIR)/

clean-cache: ## Clean only cache files
	rm -rf .pytest_cache/ cp_tox_mini/__pycache__/ tests/__pycache__/
	find . -name "*.pyc" -delete

# Environment info
env-info: ## Show environment information
	@echo "🔧 Environment Information:"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Working Directory: $(shell pwd)"
	@echo "Virtual Environment: $(VENV_DIR)"
	@echo "Random Seed: $(RANDOM_SEED)"
	@echo "Data Directory: $(DATA_DIR)"
	@echo "Reports Directory: $(REPORTS_DIR)"
