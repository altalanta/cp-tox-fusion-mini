# cp-tox-fusion-mini

A lightweight, reproducible Cell Painting × toxicity fusion workflow designed for Apple Silicon laptops. The project trims BBBC021 to a CPU-friendly subset, combines morphological profiles with Tox21 chemical descriptors, and compares image-only, chemistry-only, and late-fusion models with rigorous QC and error analysis.

## Quickstart

```bash
# create environment (once)
make env

# pull down minimal datasets (~100 images + small Tox21 slice)
make download-data

# segment, featurize, and build descriptors
make segment
make features
make qc
make descriptors

# align modalities, train models, and review reports
make fuse
make train
make error
```

Artifacts are written under `data/processed/` and `reports/`. Each CLI command gracefully reports missing prerequisites and hints at next steps.

## Repository layout

- `src/` – modular pipeline (downloaders, segmentation, feature extraction, QC, descriptor building, fusion, training, error analysis).
- `data/` – raw/interim/processed storage (kept small; `.gitkeep` placeholders only).
- `reports/` – QC summaries, model metrics, calibration plots, and the final Markdown error analysis.
- `notebooks/` – exploratory notebooks for data overview, QC review, dose–response fitting, modeling, and biological interpretation.
- `config/` – placeholder CellProfiler pipeline for future extensions.
- `tests/` – unit tests validating QC metrics, fusion logic, and model scaffolding.
- `Makefile` – task shortcuts; `env.yml` – conda environment definition.

## Data

- **Cell Painting**: curated BBBC021 (MCF7) subset with three channels (DNA/actin/tubulin) and ~12 compounds × 3–4 doses. Download via `python -m src.download_bbbc021` using hard-coded, publicly hosted TIFF URLs.
- **Tox21**: `tox21.csv.gz` slice (≤5k rows) fetched from DeepChem S3; descriptors generated with RDKit (Morgan fingerprints + 17 physchem features).

## Modeling overview

1. **Segmentation** – Cellpose (`cyto2`, deterministic seeds) produces per-field masks stored as PNG.
2. **Morphology features** – regionprops + pycytominer aggregation into per-well profiles (area, shape, intensity, texture, cell counts).
3. **QC metrics** – focus variance, illumination gradient, debris ratio, control drift; results saved as `qc_metrics.parquet` and `reports/qc_report.md` with plots.
4. **Descriptors** – RDKit Morgan (1024-bit) fingerprints + 17 physchem descriptors (`data/processed/rdkit_features.parquet`).
5. **Fusion prep** – heuristically map compound IDs (or load a CSV), normalize viability, create groupwise train/val/test splits (`data/processed/splits.json`).
6. **Models**
   - Image: shallow 3-channel CNN with a learned channel mixer; exports 128-D embeddings and logits.
  - Chem: LightGBM classifier with early stopping.
   - Fusion: concatenate mean image embeddings with PCA-reduced chem embeddings → logistic regression head.
7. **Error analysis** – calibration curves, confusion matrices, QC residuals, fusion benefit table; compiled into `reports/cp-tox-mini_report.md`.

`reports/model_metrics.json` summarizes AUROC / AP / accuracy for all models (train/val/test), while `reports/channel_weights.json` logs the learned per-channel importances.

| Model      | AUROC (test) | Average Precision (test) | Accuracy (test) |
|------------|--------------|---------------------------|-----------------|
| Image CNN  | *(see reports/model_metrics.json)* | *(see reports/model_metrics.json)* | *(see reports/model_metrics.json)* |
| Chem LGBM  | *(see reports/model_metrics.json)* | *(see reports/model_metrics.json)* | *(see reports/model_metrics.json)* |
| Late Fusion| *(see reports/model_metrics.json)* | *(see reports/model_metrics.json)* | *(see reports/model_metrics.json)* |

## Ethics and limitations

- Extremely small dataset with heuristic compound mapping → expect noisy generalization and potential label leakage; never deploy clinically.
- Fusion relies on compound ID overlap; supply a trusted mapping CSV for meaningful results.
- BBBC021 subset lacks full experimental metadata (MoA, plate layouts); treat conclusions as illustrative only.
- Apple M1 CPU-only focus means limited hyperparameter sweeps and truncated epochs.

## Uncertainty, Audit, and Temporal Validation

The pipeline now includes enhanced evaluation capabilities for uncertainty quantification, data auditing, and temporal validation:

```bash
# Run a quick audit on your dataset
python -m src.audit_temporal --data data/processed/cp_dataset.parquet --y target \
  --features "feature_*" --groups "batch_id,plate_id" --out artifacts/audit

# Evaluate with MC-Dropout + slices + calibration
python -m src.error_analysis --mc-dropout 30 --abstain-std 0.06 \
  --groups batch_id plate_id compound_id --out artifacts/eval

# Temporal evaluation (expanding window)
python -m src.error_analysis --temporal-eval --date-col assay_date \
  --mc-dropout 30 --out artifacts/eval

# Run comprehensive audit + evaluation
python scripts/run_audit_and_temporal.py --data data/processed/cp_dataset.parquet \
  --y target --features "feature_*" --groups "batch_id,plate_id,compound_id" \
  --mc-dropout 30 --out artifacts
```

**Key Features:**
- **MC-Dropout**: Provides predictive mean & std; use `--abstain-std` to defer uncertain cases
- **Data Audit**: Integrity checks, leakage detection, class balance, feature drift (KS/PSI), batch effects  
- **Temporal Validation**: Time-based splits using `assay_date` or pseudo-temporal ordering via `batch_id`
- **Slice Metrics**: Performance breakdown by groups (batch, plate, compound)
- **Enhanced Plots**: Uncertainty histograms, coverage curves, precision-recall, temporal trends

Results are saved to timestamped directories under your specified `--out` path with comprehensive JSON metrics, CSV data, and visualization plots.

## Next steps

1. Replace heuristic ID mapping with curated metadata (SMILES ↔ perturbation table).
2. Expand QC (e.g., illumination correction, plate-level batch diagnostics).
3. Port dose–response notebook (`02_dose_response_ic50.ipynb`) into a tested CLI utility.
4. Add self-contained data checks (hash validation, manifest provenance).
