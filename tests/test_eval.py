"""Tests for model evaluation functionality."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from cp_tox_mini.eval import (
    expected_calibration_error,
    compute_metrics,
    train_baseline_model,
    evaluate_model_and_save
)


class TestExpectedCalibrationError:
    """Test Expected Calibration Error calculation."""
    
    def test_perfect_calibration(self):
        """Test ECE for perfectly calibrated predictions."""
        # Perfect calibration: probabilities match actual frequencies
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        ece = expected_calibration_error(y_true, y_prob, n_bins=4)
        
        # Should be close to 0 for perfect calibration
        assert ece == pytest.approx(0.0, abs=0.1)
        
    def test_poor_calibration(self):
        """Test ECE for poorly calibrated predictions."""
        # Poor calibration: high confidence but wrong predictions
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        
        ece = expected_calibration_error(y_true, y_prob, n_bins=2)
        
        # Should be high for poor calibration
        assert ece > 0.5
        
    def test_ece_edge_cases(self):
        """Test ECE with edge cases."""
        # All same class
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.6, 0.7, 0.8, 0.9])
        
        ece = expected_calibration_error(y_true, y_prob)
        assert 0 <= ece <= 1
        
        # All probabilities in one bin
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        
        ece = expected_calibration_error(y_true, y_prob)
        assert 0 <= ece <= 1


class TestComputeMetrics:
    """Test comprehensive metrics computation."""
    
    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])  # Perfect predictions
        y_prob = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.9])
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["auroc"] == 1.0
        assert metrics["ap"] == 1.0
        
    def test_random_predictions(self):
        """Test metrics for random predictions."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_pred = np.random.binomial(1, 0.5, 100)
        y_prob = np.random.random(100)
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        # Random predictions should be around 0.5 for most metrics
        assert 0.3 <= metrics["auroc"] <= 0.7
        assert 0.2 <= metrics["accuracy"] <= 0.8
        
    def test_metrics_without_probabilities(self):
        """Test metrics computation without probabilities."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        metrics = compute_metrics(y_true, y_pred)
        
        # Should have basic metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        # Should not have probabilistic metrics
        assert "auroc" not in metrics
        assert "ap" not in metrics
        
    def test_edge_case_all_same_class(self):
        """Test metrics when all true labels are the same class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.3, 0.9])
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        # Should handle edge case gracefully
        assert 0 <= metrics["accuracy"] <= 1
        # AUROC might be undefined, but function should handle it


class TestTrainBaselineModel:
    """Test baseline model training."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        # Create some signal
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        return X, y
        
    def test_logistic_regression_training(self, sample_data):
        """Test logistic regression training."""
        X, y = sample_data
        
        model, metrics = train_baseline_model(X, y, model_type="logistic", random_state=42)
        
        # Check model type
        assert isinstance(model, LogisticRegression)
        
        # Check metrics structure
        required_metrics = ["auroc", "ap", "accuracy", "brier", "ece", "n_train", "n_test"]
        for metric in required_metrics:
            assert metric in metrics
            
        # Check reasonable performance (should be better than random)
        assert metrics["auroc"] > 0.6
        assert 0 <= metrics["accuracy"] <= 1
        
    def test_random_forest_training(self, sample_data):
        """Test random forest training."""
        X, y = sample_data
        
        model, metrics = train_baseline_model(X, y, model_type="rf", random_state=42)
        
        # Check model type
        assert isinstance(model, RandomForestClassifier)
        
        # Check metrics
        assert metrics["auroc"] > 0.6
        assert "n_features" in metrics
        assert metrics["n_features"] == X.shape[1]
        
    def test_invalid_model_type(self, sample_data):
        """Test error handling for invalid model type."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Unknown model type"):
            train_baseline_model(X, y, model_type="invalid")
            
    def test_small_dataset(self):
        """Test training with very small dataset."""
        # Minimal dataset
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        model, metrics = train_baseline_model(X, y, random_state=42)
        
        # Should complete without error
        assert "auroc" in metrics
        assert metrics["n_train"] > 0
        assert metrics["n_test"] > 0


class TestEvaluateModelAndSave:
    """Test complete model evaluation and saving."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n_samples = 80
        n_features = 15
        
        # Generate features
        features = np.random.randn(n_samples, n_features)
        feature_names = [f"feature_{i:03d}" for i in range(n_features)]
        
        # Create DataFrame
        data = pd.DataFrame(features, columns=feature_names)
        
        # Add metadata
        data["compound_id"] = [f"COMPOUND_{i%10:03d}" for i in range(n_samples)]
        data["plate_id"] = [f"plate_{(i//20)%3 + 1:03d}" for i in range(n_samples)]
        
        # Create target with signal
        data["target"] = (features[:, 0] + features[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        return data
        
    def test_evaluate_and_save_logistic(self, sample_dataframe, tmp_path):
        """Test evaluation and saving with logistic regression."""
        metrics = evaluate_model_and_save(
            sample_dataframe,
            target_col="target",
            model_type="logistic",
            output_dir=tmp_path,
            random_state=42
        )
        
        # Check returned metrics
        assert isinstance(metrics, dict)
        assert "auroc" in metrics
        assert "ap" in metrics
        assert metrics["auroc"] > 0.5  # Should be better than random
        
        # Check saved files
        assert (tmp_path / "model_metrics.json").exists()
        assert (tmp_path / "figures" / "roc.png").exists()
        assert (tmp_path / "figures" / "pr.png").exists()
        assert (tmp_path / "figures" / "calibration.png").exists()
        
    def test_evaluate_missing_target(self, sample_dataframe, tmp_path):
        """Test error handling for missing target column."""
        with pytest.raises(ValueError, match="Target column 'missing' not found"):
            evaluate_model_and_save(
                sample_dataframe,
                target_col="missing",
                output_dir=tmp_path
            )
            
    def test_evaluate_no_features(self, tmp_path):
        """Test error handling when no feature columns found."""
        # DataFrame with only metadata
        data = pd.DataFrame({
            "compound_id": ["A", "B", "C"],
            "target": [0, 1, 0]
        })
        
        with pytest.raises(ValueError, match="No feature columns found"):
            evaluate_model_and_save(data, output_dir=tmp_path)
            
    def test_evaluate_insufficient_data(self, tmp_path):
        """Test error handling with insufficient clean data."""
        # Very small dataset
        data = pd.DataFrame({
            "feature_001": [1, 2],
            "target": [0, 1]
        })
        
        with pytest.raises(ValueError, match="Insufficient clean data"):
            evaluate_model_and_save(data, output_dir=tmp_path)
            
    def test_custom_feature_columns(self, sample_dataframe, tmp_path):
        """Test evaluation with custom feature column specification."""
        # Use only subset of features
        custom_features = ["feature_000", "feature_001", "feature_002"]
        
        metrics = evaluate_model_and_save(
            sample_dataframe,
            target_col="target",
            feature_cols=custom_features,
            output_dir=tmp_path,
            random_state=42
        )
        
        assert "auroc" in metrics
        # Check that it used the specified number of features
        # (This would be in the model object, but we can at least verify it ran)
        
    def test_missing_values_handling(self, sample_dataframe, tmp_path):
        """Test handling of missing values in data."""
        # Add some missing values
        sample_dataframe.loc[0:5, "feature_000"] = np.nan
        sample_dataframe.loc[10:15, "target"] = np.nan
        
        metrics = evaluate_model_and_save(
            sample_dataframe,
            target_col="target",
            output_dir=tmp_path,
            random_state=42
        )
        
        # Should handle missing values gracefully
        assert "auroc" in metrics
        assert metrics["n_train"] + metrics["n_test"] < len(sample_dataframe)  # Some rows dropped


def test_evaluation_reproducibility(tmp_path):
    """Test that evaluation results are reproducible."""
    # Create identical datasets
    np.random.seed(42)
    n_samples = 60
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    data1 = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    data1["target"] = y
    
    data2 = data1.copy()
    
    # Run evaluation twice
    metrics1 = evaluate_model_and_save(data1, output_dir=tmp_path / "run1", random_state=42)
    metrics2 = evaluate_model_and_save(data2, output_dir=tmp_path / "run2", random_state=42)
    
    # Results should be identical
    assert metrics1["auroc"] == pytest.approx(metrics2["auroc"], abs=1e-10)
    assert metrics1["ap"] == pytest.approx(metrics2["ap"], abs=1e-10)
    assert metrics1["accuracy"] == pytest.approx(metrics2["accuracy"], abs=1e-10)