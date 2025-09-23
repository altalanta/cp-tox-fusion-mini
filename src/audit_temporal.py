"""Data audit and temporal validation utilities for cp-tox-fusion-mini."""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .utils import configure_logging, ensure_dir, save_json

LOGGER = logging.getLogger("cp_tox.audit_temporal")

# Suppress scipy warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


def run_audit(
    df: pd.DataFrame,
    y_col: str,
    feature_cols: List[str],
    group_cols: List[str] = None,
    train_mask: Optional[np.ndarray] = None,
    val_mask: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """Comprehensive data audit for integrity, balance, leakage, and drift.
    
    Args:
        df: Input dataframe
        y_col: Target column name
        feature_cols: List of feature column names
        group_cols: List of grouping columns (batch_id, plate_id, etc.)
        train_mask: Boolean mask for training data
        val_mask: Boolean mask for validation data
        output_dir: Directory to save audit artifacts
        
    Returns:
        Dictionary with audit results
    """
    if group_cols is None:
        group_cols = []
    
    LOGGER.info(f"Running audit on {len(df)} samples with {len(feature_cols)} features")
    
    audit_results = {}
    
    # === DATA INTEGRITY ===
    integrity_results = {}
    
    # Missing data
    missing_rates = df[feature_cols].isnull().mean()
    integrity_results["missing_rates"] = {
        "mean": float(missing_rates.mean()),
        "max": float(missing_rates.max()),
        "features_with_missing": missing_rates[missing_rates > 0].to_dict()
    }
    
    # Constant columns
    constant_cols = []
    for col in feature_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    integrity_results["constant_columns"] = constant_cols
    
    # Duplicate rows
    duplicate_rate = df.duplicated().mean()
    integrity_results["duplicate_rate"] = float(duplicate_rate)
    
    audit_results["integrity"] = integrity_results
    
    # === CLASS BALANCE ===
    balance_results = {}
    
    y_counts = df[y_col].value_counts()
    balance_results["class_counts"] = y_counts.to_dict()
    balance_results["class_ratio"] = float(y_counts.min() / y_counts.max())
    
    # Per-group balance if group columns exist
    if group_cols:
        group_balance = {}
        for group_col in group_cols:
            if group_col in df.columns:
                group_stats = df.groupby(group_col)[y_col].agg(['count', 'mean']).to_dict()
                group_balance[group_col] = group_stats
        balance_results["per_group_balance"] = group_balance
    
    audit_results["balance"] = balance_results
    
    # === LEAKAGE DETECTION ===
    leakage_results = {}
    
    # Feature-target correlations
    correlations = {}
    high_corr_features = []
    
    for col in feature_cols:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            try:
                corr, p_val = stats.spearmanr(df[col].dropna(), df.loc[df[col].notna(), y_col])
                correlations[col] = {"correlation": float(corr), "p_value": float(p_val)}
                
                if abs(corr) > 0.8:  # Very high correlation
                    high_corr_features.append(col)
            except Exception:
                correlations[col] = {"correlation": np.nan, "p_value": np.nan}
    
    leakage_results["feature_target_correlations"] = correlations
    leakage_results["high_correlation_features"] = high_corr_features
    
    # Extremely predictive single features
    highly_predictive = []
    if len(np.unique(df[y_col])) > 1:  # Need both classes
        for col in feature_cols[:50]:  # Limit to avoid long computation
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32'] and df[col].notna().sum() > 10:
                try:
                    feature_vals = df[col].fillna(df[col].median()).values.reshape(-1, 1)
                    auc = roc_auc_score(df[y_col], feature_vals.flatten())
                    auc = max(auc, 1 - auc)  # Take max of AUC and 1-AUC
                    
                    if auc > 0.9:
                        highly_predictive.append({"feature": col, "auc": float(auc)})
                except Exception:
                    continue
    
    leakage_results["highly_predictive_features"] = highly_predictive
    
    # Train/val correlation differences (if splits provided)
    if train_mask is not None and val_mask is not None:
        corr_diffs = {}
        for col in feature_cols[:30]:  # Limit computation
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                try:
                    train_corr, _ = stats.spearmanr(
                        df.loc[train_mask, col].dropna(),
                        df.loc[train_mask & df[col].notna(), y_col]
                    )
                    val_corr, _ = stats.spearmanr(
                        df.loc[val_mask, col].dropna(),
                        df.loc[val_mask & df[col].notna(), y_col]
                    )
                    diff = abs(train_corr - val_corr)
                    if diff > 0.2:  # Flag large differences
                        corr_diffs[col] = {
                            "train_corr": float(train_corr),
                            "val_corr": float(val_corr),
                            "diff": float(diff)
                        }
                except Exception:
                    continue
        
        leakage_results["train_val_correlation_diffs"] = corr_diffs
    
    audit_results["leakage"] = leakage_results
    
    # === FEATURE DRIFT ===
    drift_results = {}
    
    # KS tests across groups
    if group_cols and train_mask is not None and val_mask is not None:
        ks_results = {}
        for col in feature_cols[:20]:  # Limit to avoid long computation
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                try:
                    train_vals = df.loc[train_mask, col].dropna()
                    val_vals = df.loc[val_mask, col].dropna()
                    
                    if len(train_vals) > 10 and len(val_vals) > 10:
                        ks_stat, p_val = stats.ks_2samp(train_vals, val_vals)
                        ks_results[col] = {"ks_stat": float(ks_stat), "p_value": float(p_val)}
                except Exception:
                    continue
        
        # Sort by KS statistic and take top drifting features
        sorted_ks = sorted(ks_results.items(), key=lambda x: x[1]["ks_stat"], reverse=True)
        drift_results["ks_tests"] = dict(sorted_ks[:10])
    
    audit_results["drift"] = drift_results
    
    # === BATCH EFFECTS ===
    batch_results = {}
    
    for batch_col in ["batch_id", "plate_id"]:
        if batch_col in df.columns and batch_col in group_cols:
            batch_effects = {}
            for col in feature_cols[:10]:  # Top features only
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    try:
                        groups = [group[col].dropna() for name, group in df.groupby(batch_col)]
                        groups = [g for g in groups if len(g) > 3]  # Need sufficient samples
                        
                        if len(groups) > 1:
                            f_stat, p_val = stats.f_oneway(*groups)
                            batch_effects[col] = {"f_stat": float(f_stat), "p_value": float(p_val)}
                    except Exception:
                        continue
            
            if batch_effects:
                batch_results[batch_col] = batch_effects
    
    audit_results["batch_effects"] = batch_results
    
    # === SAVE RESULTS ===
    if output_dir:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        # Save JSON
        save_json(audit_results, output_dir / "audit_results.json")
        
        # Save compact Markdown report
        _save_audit_report(audit_results, output_dir / "audit_report.md")
        
        LOGGER.info(f"Audit results saved to {output_dir}")
    
    return audit_results


def _save_audit_report(results: Dict, output_path: Path) -> None:
    """Save a compact Markdown audit report."""
    with open(output_path, "w") as f:
        f.write("# Data Audit Report\n\n")
        
        # Integrity
        integrity = results.get("integrity", {})
        f.write("## Data Integrity\n\n")
        f.write(f"- **Missing data**: {integrity.get('missing_rates', {}).get('mean', 0):.1%} average, "
                f"{integrity.get('missing_rates', {}).get('max', 0):.1%} maximum\n")
        f.write(f"- **Constant features**: {len(integrity.get('constant_columns', []))}\n")
        f.write(f"- **Duplicate rows**: {integrity.get('duplicate_rate', 0):.1%}\n\n")
        
        # Balance
        balance = results.get("balance", {})
        f.write("## Class Balance\n\n")
        class_counts = balance.get("class_counts", {})
        for label, count in class_counts.items():
            f.write(f"- **Class {label}**: {count} samples\n")
        f.write(f"- **Class ratio**: {balance.get('class_ratio', 0):.3f}\n\n")
        
        # Leakage
        leakage = results.get("leakage", {})
        f.write("## Leakage Detection\n\n")
        f.write(f"- **High correlation features**: {len(leakage.get('high_correlation_features', []))}\n")
        f.write(f"- **Highly predictive features**: {len(leakage.get('highly_predictive_features', []))}\n")
        
        highly_pred = leakage.get('highly_predictive_features', [])
        if highly_pred:
            f.write("- **Top predictive features**:\n")
            for feat in highly_pred[:3]:
                f.write(f"  - {feat['feature']}: AUC = {feat['auc']:.3f}\n")
        f.write("\n")
        
        # Drift
        drift = results.get("drift", {})
        ks_tests = drift.get("ks_tests", {})
        f.write("## Feature Drift\n\n")
        f.write(f"- **Features with significant drift**: {len(ks_tests)}\n")
        if ks_tests:
            f.write("- **Top drifting features**:\n")
            for feat, stats in list(ks_tests.items())[:3]:
                f.write(f"  - {feat}: KS = {stats['ks_stat']:.3f}, p = {stats['p_value']:.3e}\n")
        f.write("\n")
        
        # Batch effects
        batch = results.get("batch_effects", {})
        f.write("## Batch Effects\n\n")
        for batch_col, effects in batch.items():
            f.write(f"- **{batch_col}**: {len(effects)} features with significant batch effects\n")


def has_assay_date(df: pd.DataFrame, date_col: str = "assay_date") -> bool:
    """Check if dataframe has a parseable date column.
    
    Args:
        df: Input dataframe
        date_col: Name of date column to check
        
    Returns:
        True if date column exists and is parseable
    """
    if date_col not in df.columns:
        return False
    
    try:
        pd.to_datetime(df[date_col].dropna().iloc[:10])  # Test first 10 non-null values
        return True
    except Exception:
        return False


def build_temporal_folds(
    df: pd.DataFrame,
    date_col: str = "assay_date",
    n_folds: int = 3,
    mode: str = "expanding",
    batch_col: Optional[str] = "batch_id"
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build temporal cross-validation folds with strict time ordering.
    
    Args:
        df: Input dataframe
        date_col: Date column name (if exists)
        n_folds: Number of folds to create
        mode: "expanding" (expanding window) or "rolling" (rolling blocks)
        batch_col: Fallback batch column for ordering
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    LOGGER.info(f"Building {n_folds} temporal folds in {mode} mode")
    
    # Determine temporal ordering
    if has_assay_date(df, date_col):
        LOGGER.info(f"Using {date_col} for temporal ordering")
        df_sorted = df.copy()
        df_sorted["_temporal_order"] = pd.to_datetime(df[date_col])
        df_sorted = df_sorted.sort_values("_temporal_order")
        order_indices = df_sorted.index.values
    elif batch_col and batch_col in df.columns:
        LOGGER.info(f"Using {batch_col} for pseudo-temporal ordering")
        df_sorted = df.copy()
        # Group by batch and use group order as proxy for time
        batch_order = df.groupby(batch_col).first().index.sort_values()
        batch_to_order = {batch: i for i, batch in enumerate(batch_order)}
        df_sorted["_temporal_order"] = df_sorted[batch_col].map(batch_to_order)
        df_sorted = df_sorted.sort_values("_temporal_order")
        order_indices = df_sorted.index.values
    else:
        LOGGER.warning("No date or batch column found, using row order")
        order_indices = df.index.values
    
    n_samples = len(order_indices)
    folds = []
    
    if mode == "expanding":
        # Expanding window: train on [0, k], test on [k+1, k+step]
        for i in range(n_folds):
            # Split into roughly equal test sets
            test_start = int(n_samples * (i + 1) / (n_folds + 1))
            test_end = int(n_samples * (i + 2) / (n_folds + 1))
            
            train_indices = order_indices[:test_start]
            test_indices = order_indices[test_start:test_end]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                folds.append((train_indices, test_indices))
    
    elif mode == "rolling":
        # Rolling blocks: train on [k, k+window], test on [k+window+1, k+2*window]
        window_size = n_samples // (n_folds * 2)
        
        for i in range(n_folds):
            start_idx = i * window_size
            mid_idx = start_idx + window_size
            end_idx = mid_idx + window_size
            
            if end_idx <= n_samples:
                train_indices = order_indices[start_idx:mid_idx]
                test_indices = order_indices[mid_idx:end_idx]
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    folds.append((train_indices, test_indices))
    
    LOGGER.info(f"Created {len(folds)} temporal folds")
    
    # Validate temporal ordering
    for i, (train_idx, test_idx) in enumerate(folds):
        # Check that all training data comes before test data
        if has_assay_date(df, date_col):
            train_dates = pd.to_datetime(df.loc[train_idx, date_col]).max()
            test_dates = pd.to_datetime(df.loc[test_idx, date_col]).min()
            
            if train_dates >= test_dates:
                LOGGER.warning(f"Fold {i}: temporal ordering violation detected")
        
        LOGGER.info(f"Fold {i}: {len(train_idx)} train, {len(test_idx)} test samples")
    
    return folds


def compute_temporal_metrics(
    folds: List[Tuple[np.ndarray, np.ndarray]],
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> pd.DataFrame:
    """Compute metrics across temporal folds.
    
    Args:
        folds: List of (train_indices, test_indices) tuples
        predictions: Dict mapping fold_id to predictions for that fold
        y_true: True labels
        
    Returns:
        DataFrame with metrics per fold
    """
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    
    metrics_list = []
    
    for i, (train_idx, test_idx) in enumerate(folds):
        fold_id = f"fold_{i}"
        
        if fold_id not in predictions:
            continue
        
        y_test = y_true[test_idx]
        y_pred_proba = predictions[fold_id]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        if len(np.unique(y_test)) > 1:  # Need both classes for AUC
            auroc = roc_auc_score(y_test, y_pred_proba)
            auprc = average_precision_score(y_test, y_pred_proba)
        else:
            auroc = np.nan
            auprc = np.nan
        
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics_list.append({
            "fold": i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": accuracy
        })
    
    return pd.DataFrame(metrics_list)


def main():
    """CLI for running data audit."""
    parser = argparse.ArgumentParser(description="Run data audit for cp-tox-fusion-mini")
    parser.add_argument("--data", required=True, help="Path to data CSV/parquet file")
    parser.add_argument("--y", required=True, help="Target column name")
    parser.add_argument("--features", required=True, help="Feature column pattern (e.g., 'x_*')")
    parser.add_argument("--groups", help="Comma-separated group columns (e.g., 'batch_id,plate_id')")
    parser.add_argument("--out", default="artifacts/audit", help="Output directory")
    
    args = parser.parse_args()
    
    configure_logging()
    
    # Load data
    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    LOGGER.info(f"Loaded {len(df)} samples from {data_path}")
    
    # Parse feature columns
    if "*" in args.features:
        feature_pattern = args.features.replace("*", "")
        feature_cols = [col for col in df.columns if col.startswith(feature_pattern)]
    else:
        feature_cols = [args.features]
    
    # Parse group columns
    group_cols = []
    if args.groups:
        group_cols = [col.strip() for col in args.groups.split(",")]
    
    # Run audit
    output_dir = Path(args.out)
    audit_results = run_audit(
        df=df,
        y_col=args.y,
        feature_cols=feature_cols,
        group_cols=group_cols,
        output_dir=output_dir
    )
    
    LOGGER.info("Audit complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()