"""Tests for temporal validation and data audit utilities."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.audit_temporal import (
    build_temporal_folds,
    compute_temporal_metrics,
    has_assay_date,
    run_audit,
)


@pytest.fixture
def sample_temporal_data():
    """Create sample data with temporal information."""
    np.random.seed(42)
    n = 200
    
    # Create dates spanning 6 months
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i*3) for i in range(n)]
    
    df = pd.DataFrame({
        'assay_date': dates,
        'batch_id': [f'batch_{i//20}' for i in range(n)],  # 10 batches
        'compound_id': [f'compound_{i%15}' for i in range(n)],  # 15 compounds
        'well_id': [f'well_{i}' for i in range(n)],
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'target': np.random.binomial(1, 0.3, n)
    })
    
    return df


@pytest.fixture
def sample_data_no_date():
    """Create sample data without date column."""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'batch_id': [f'batch_{i//10}' for i in range(n)],  # 10 batches
        'compound_id': [f'compound_{i%8}' for i in range(n)],
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'target': np.random.binomial(1, 0.4, n)
    })
    
    return df


class TestHasAssayDate:
    """Test date column detection."""
    
    def test_has_assay_date_valid(self, sample_temporal_data):
        """Test detection of valid date column."""
        assert has_assay_date(sample_temporal_data, "assay_date")
    
    def test_has_assay_date_missing_column(self, sample_data_no_date):
        """Test with missing date column."""
        assert not has_assay_date(sample_data_no_date, "assay_date")
    
    def test_has_assay_date_invalid_format(self):
        """Test with invalid date format."""
        df = pd.DataFrame({
            'assay_date': ['not_a_date', 'also_not_a_date', '2023-13-45'],
            'feature': [1, 2, 3]
        })
        assert not has_assay_date(df, "assay_date")
    
    def test_has_assay_date_mixed_format(self):
        """Test with mixed valid/invalid dates."""
        df = pd.DataFrame({
            'assay_date': ['2023-01-01', '2023-01-02', 'invalid_date'],
            'feature': [1, 2, 3]
        })
        # Should still return True if first few are parseable
        assert has_assay_date(df, "assay_date")


class TestBuildTemporalFolds:
    """Test temporal fold creation."""
    
    def test_temporal_folds_expanding_with_dates(self, sample_temporal_data):
        """Test expanding window with date column."""
        folds = build_temporal_folds(
            sample_temporal_data, 
            date_col="assay_date", 
            n_folds=3, 
            mode="expanding"
        )
        
        assert len(folds) == 3
        
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap
            
            # Check temporal ordering
            train_dates = sample_temporal_data.loc[train_idx, "assay_date"]
            test_dates = sample_temporal_data.loc[test_idx, "assay_date"]
            
            # All training dates should be before all test dates
            assert train_dates.max() < test_dates.min()
    
    def test_temporal_folds_rolling_with_dates(self, sample_temporal_data):
        """Test rolling window with date column."""
        folds = build_temporal_folds(
            sample_temporal_data,
            date_col="assay_date",
            n_folds=3,
            mode="rolling"
        )
        
        assert len(folds) >= 1  # Should produce at least 1 fold
        
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            
            # Check temporal ordering for rolling windows
            train_dates = sample_temporal_data.loc[train_idx, "assay_date"]
            test_dates = sample_temporal_data.loc[test_idx, "assay_date"]
            
            # Training dates should come before test dates
            assert train_dates.max() <= test_dates.min()
    
    def test_temporal_folds_without_dates(self, sample_data_no_date):
        """Test temporal folds using batch_id as proxy."""
        folds = build_temporal_folds(
            sample_data_no_date,
            date_col="assay_date",  # Doesn't exist
            n_folds=3,
            batch_col="batch_id"
        )
        
        assert len(folds) >= 1
        
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap
    
    def test_temporal_folds_no_date_no_batch(self):
        """Test fallback to row order."""
        df = pd.DataFrame({
            'feature': np.random.randn(50),
            'target': np.random.binomial(1, 0.5, 50)
        })
        
        folds = build_temporal_folds(df, n_folds=2)
        
        assert len(folds) >= 1
        
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # With row order, train indices should come before test indices
            assert np.max(train_idx) < np.min(test_idx)
    
    def test_temporal_folds_small_dataset(self):
        """Test with very small dataset."""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        
        folds = build_temporal_folds(df, n_folds=2)
        
        # Should handle small datasets gracefully
        assert isinstance(folds, list)
        
        if folds:  # If any folds are created
            for train_idx, test_idx in folds:
                assert len(train_idx) > 0
                assert len(test_idx) > 0


class TestComputeTemporalMetrics:
    """Test temporal metrics computation."""
    
    def test_compute_temporal_metrics(self):
        """Test temporal metrics computation."""
        # Create mock fold data
        folds = [
            (np.array([0, 1, 2, 3]), np.array([4, 5, 6])),
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8, 9])),
            (np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([8, 9, 10, 11]))
        ]
        
        # Create mock predictions
        np.random.seed(42)
        predictions = {}
        for i in range(len(folds)):
            n_test = len(folds[i][1])
            predictions[f"fold_{i}"] = np.random.random(n_test)
        
        # True labels
        y_true = np.random.binomial(1, 0.3, 12)
        
        metrics_df = compute_temporal_metrics(folds, predictions, y_true)
        
        assert isinstance(metrics_df, pd.DataFrame)
        assert len(metrics_df) == len(folds)
        assert "fold" in metrics_df.columns
        assert "n_train" in metrics_df.columns
        assert "n_test" in metrics_df.columns
        assert "auroc" in metrics_df.columns
        assert "auprc" in metrics_df.columns
        assert "accuracy" in metrics_df.columns
        
        # Check that fold numbers are correct
        assert list(metrics_df["fold"]) == [0, 1, 2]
    
    def test_compute_temporal_metrics_missing_predictions(self):
        """Test with missing predictions for some folds."""
        folds = [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([0, 1, 2]), np.array([3, 4, 5]))
        ]
        
        # Only provide predictions for first fold
        predictions = {"fold_0": np.array([0.6, 0.8])}
        y_true = np.array([0, 1, 0, 1, 1, 0])
        
        metrics_df = compute_temporal_metrics(folds, predictions, y_true)
        
        # Should only have metrics for fold with predictions
        assert len(metrics_df) == 1
        assert metrics_df.iloc[0]["fold"] == 0


class TestRunAudit:
    """Test comprehensive data audit functionality."""
    
    def test_run_audit_basic(self, sample_temporal_data):
        """Test basic audit functionality."""
        feature_cols = ["feature_1", "feature_2", "feature_3"]
        group_cols = ["batch_id", "compound_id"]
        
        audit_results = run_audit(
            df=sample_temporal_data,
            y_col="target",
            feature_cols=feature_cols,
            group_cols=group_cols
        )
        
        assert isinstance(audit_results, dict)
        assert "integrity" in audit_results
        assert "balance" in audit_results
        assert "leakage" in audit_results
        assert "drift" in audit_results
        assert "batch_effects" in audit_results
    
    def test_audit_integrity_checks(self):
        """Test data integrity checks."""
        # Create data with various integrity issues
        df = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5],  # Missing value
            'feature_2': [1, 1, 1, 1, 1],  # Constant
            'feature_3': [1, 2, 3, 4, 5],  # Normal
            'target': [0, 1, 0, 1, 0]
        })
        # Add duplicate row
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        
        audit_results = run_audit(
            df=df,
            y_col="target",
            feature_cols=["feature_1", "feature_2", "feature_3"]
        )
        
        integrity = audit_results["integrity"]
        
        # Check missing data detection
        assert integrity["missing_rates"]["max"] > 0  # feature_1 has missing
        
        # Check constant column detection
        assert "feature_2" in integrity["constant_columns"]
        
        # Check duplicate detection
        assert integrity["duplicate_rate"] > 0
    
    def test_audit_class_balance(self):
        """Test class balance analysis."""
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'batch_id': [f'batch_{i//20}' for i in range(100)],
            'target': [1] * 80 + [0] * 20  # Imbalanced
        })
        
        audit_results = run_audit(
            df=df,
            y_col="target",
            feature_cols=["feature_1"],
            group_cols=["batch_id"]
        )
        
        balance = audit_results["balance"]
        
        # Check class counts
        assert balance["class_counts"][0] == 20
        assert balance["class_counts"][1] == 80
        
        # Check class ratio (min/max)
        assert balance["class_ratio"] == 20/80  # 0.25
        
        # Check per-group balance
        assert "per_group_balance" in balance
        assert "batch_id" in balance["per_group_balance"]
    
    def test_audit_with_train_val_split(self, sample_temporal_data):
        """Test audit with train/validation splits."""
        n = len(sample_temporal_data)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:n//2] = True
        val_mask = np.zeros(n, dtype=bool) 
        val_mask[n//2:] = True
        
        audit_results = run_audit(
            df=sample_temporal_data,
            y_col="target",
            feature_cols=["feature_1", "feature_2"],
            train_mask=train_mask,
            val_mask=val_mask
        )
        
        # Should include train/val correlation differences
        leakage = audit_results["leakage"]
        assert "train_val_correlation_diffs" in leakage


class TestTemporalSplitValidation:
    """Test temporal split validation properties."""
    
    def test_no_leakage_expanding(self, sample_temporal_data):
        """Test that expanding window prevents data leakage."""
        folds = build_temporal_folds(
            sample_temporal_data,
            date_col="assay_date",
            n_folds=3,
            mode="expanding"
        )
        
        # Check each fold for temporal ordering
        for i, (train_idx, test_idx) in enumerate(folds):
            train_dates = sample_temporal_data.loc[train_idx, "assay_date"]
            test_dates = sample_temporal_data.loc[test_idx, "assay_date"]
            
            # No train date should be >= any test date
            latest_train = train_dates.max()
            earliest_test = test_dates.min()
            
            assert latest_train < earliest_test, f"Fold {i}: temporal leakage detected"
    
    def test_monotonic_train_size_expanding(self, sample_temporal_data):
        """Test that training size increases in expanding window."""
        folds = build_temporal_folds(
            sample_temporal_data,
            date_col="assay_date",
            n_folds=4,
            mode="expanding"
        )
        
        train_sizes = [len(train_idx) for train_idx, _ in folds]
        
        # Training size should be non-decreasing
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1], "Training size should increase in expanding window"
    
    def test_coverage_completeness(self, sample_temporal_data):
        """Test that folds cover all data points."""
        folds = build_temporal_folds(
            sample_temporal_data,
            date_col="assay_date", 
            n_folds=3,
            mode="expanding"
        )
        
        all_indices = set()
        for train_idx, test_idx in folds:
            all_indices.update(train_idx)
            all_indices.update(test_idx)
        
        # Should cover most of the dataset
        coverage = len(all_indices) / len(sample_temporal_data)
        assert coverage > 0.8, f"Folds only cover {coverage:.1%} of data"
    
    def test_fold_size_reasonableness(self, sample_temporal_data):
        """Test that fold sizes are reasonable."""
        folds = build_temporal_folds(
            sample_temporal_data,
            date_col="assay_date",
            n_folds=3,
            mode="expanding"
        )
        
        for i, (train_idx, test_idx) in enumerate(folds):
            # Each fold should have reasonable number of samples
            assert len(train_idx) >= 10, f"Fold {i}: training set too small ({len(train_idx)})"
            assert len(test_idx) >= 5, f"Fold {i}: test set too small ({len(test_idx)})"
            
            # Test set shouldn't be too large relative to training
            ratio = len(test_idx) / len(train_idx)
            assert ratio <= 2.0, f"Fold {i}: test set too large relative to training ({ratio:.2f})"