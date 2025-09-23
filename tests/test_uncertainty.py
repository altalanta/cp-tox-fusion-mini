"""Tests for uncertainty quantification utilities."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.uncertainty import (
    apply_abstention,
    compute_coverage_vs_performance,
    detect_model_dropout_layers,
    enable_dropout,
    mc_predict,
    uncertainty_histogram_data,
)


class ToyMLP(nn.Module):
    """Simple MLP with dropout for testing."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layers(x).squeeze(-1)


@pytest.fixture
def toy_model():
    """Create a toy model with dropout."""
    torch.manual_seed(42)
    return ToyMLP()


@pytest.fixture
def toy_data():
    """Create toy dataset."""
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    return loader


class TestDropoutDetection:
    """Test dropout layer detection."""
    
    def test_detect_dropout_layers(self, toy_model):
        """Test counting dropout layers."""
        count = detect_model_dropout_layers(toy_model)
        assert count == 2  # ToyMLP has 2 dropout layers
    
    def test_no_dropout_model(self):
        """Test model without dropout."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        count = detect_model_dropout_layers(model)
        assert count == 0


class TestEnableDropout:
    """Test selective dropout enabling."""
    
    def test_enable_dropout(self, toy_model):
        """Test that enable_dropout sets only dropout layers to train mode."""
        toy_model.eval()  # Set all to eval mode
        
        # Verify all modules are in eval mode initially
        for module in toy_model.modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                assert not module.training
            elif isinstance(module, nn.Dropout):
                assert not module.training
        
        # Enable dropout
        enable_dropout(toy_model)
        
        # Check that dropout layers are in train mode, others in eval
        for module in toy_model.modules():
            if isinstance(module, nn.Dropout):
                assert module.training
            elif isinstance(module, (nn.Linear, nn.ReLU)):
                assert not module.training


class TestMCPredict:
    """Test MC-Dropout prediction."""
    
    def test_mc_predict_shapes(self, toy_model, toy_data):
        """Test that MC-Dropout returns correct shapes."""
        results = mc_predict(toy_model, toy_data, T=5)
        
        expected_n = 100  # Number of samples in toy_data
        
        assert "mean_prob" in results
        assert "std_prob" in results
        assert "entropy" in results
        assert "probs_T" in results
        
        assert results["mean_prob"].shape == (expected_n,)
        assert results["std_prob"].shape == (expected_n,)
        assert results["entropy"].shape == (expected_n,)
        assert results["probs_T"].shape == (5, expected_n)
    
    def test_mc_predict_variance(self, toy_model, toy_data):
        """Test that MC-Dropout produces non-zero variance."""
        results = mc_predict(toy_model, toy_data, T=20)
        
        # With dropout, we should see some variance
        assert np.mean(results["std_prob"]) > 0
        assert np.max(results["std_prob"]) > 0.001  # Some meaningful variance
    
    def test_mc_predict_deterministic_without_dropout(self, toy_data):
        """Test that model without dropout produces zero variance."""
        model_no_dropout = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        results = mc_predict(model_no_dropout, toy_data, T=10)
        
        # Without dropout, variance should be exactly zero
        assert np.allclose(results["std_prob"], 0, atol=1e-8)
    
    def test_temperature_scaling(self, toy_model, toy_data):
        """Test temperature scaling effect."""
        results_normal = mc_predict(toy_model, toy_data, T=5)
        results_scaled = mc_predict(toy_model, toy_data, T=5, temperature=2.0)
        
        # Temperature scaling should affect probabilities
        assert not np.allclose(results_normal["mean_prob"], results_scaled["mean_prob"])
        
        # Higher temperature should generally produce probabilities closer to 0.5
        normal_extreme = np.mean(np.abs(results_normal["mean_prob"] - 0.5))
        scaled_extreme = np.mean(np.abs(results_scaled["mean_prob"] - 0.5))
        assert scaled_extreme <= normal_extreme


class TestAbstention:
    """Test abstention functionality."""
    
def setup_abstention_data(self):
        """Create test data for abstention."""
        np.random.seed(42)
        n = 100
        mean_prob = np.random.beta(2, 2, n)  # Varied probabilities
        std_prob = np.random.exponential(0.1, n)  # Some with high uncertainty
        entropy = np.random.exponential(0.2, n)
        return mean_prob, std_prob, entropy
    
    def test_apply_abstention_std(self):
        """Test abstention based on standard deviation."""
        mean_prob, std_prob, entropy = self.setup_abstention_data()
        
        results = apply_abstention(
            mean_prob, std_prob, 
            thresh=0.5, std_cut=0.15
        )
        
        assert "predictions" in results
        assert "abstained_mask" in results
        assert "confident_mask" in results
        
        # Check that high uncertainty samples are abstained
        high_std_mask = std_prob > 0.15
        assert np.all(results["abstained_mask"][high_std_mask])
        
        # Check that confident predictions are made
        confident_samples = results["confident_mask"]
        confident_preds = results["predictions"][confident_samples]
        assert np.all((confident_preds == 0) | (confident_preds == 1))
        
        # Check that abstained samples have prediction -1
        abstained_preds = results["predictions"][results["abstained_mask"]]
        assert np.all(abstained_preds == -1)
    
    def test_apply_abstention_entropy(self):
        """Test abstention based on entropy."""
        mean_prob, std_prob, entropy = self.setup_abstention_data()
        
        results = apply_abstention(
            mean_prob, std_prob,
            thresh=0.5, ent_cut=0.3, entropy=entropy
        )
        
        # Check that high entropy samples are abstained
        high_ent_mask = entropy > 0.3
        assert np.all(results["abstained_mask"][high_ent_mask])
    
    def test_apply_abstention_combined(self):
        """Test abstention with both std and entropy thresholds."""
        mean_prob, std_prob, entropy = self.setup_abstention_data()
        
        results = apply_abstention(
            mean_prob, std_prob,
            thresh=0.5, std_cut=0.15, ent_cut=0.3, entropy=entropy
        )
        
        # Should abstain if EITHER condition is met
        expected_abstained = (std_prob > 0.15) | (entropy > 0.3)
        assert np.array_equal(results["abstained_mask"], expected_abstained)


class TestCoverageVsPerformance:
    """Test coverage vs performance analysis."""
    
    def test_compute_coverage_vs_performance(self):
        """Test coverage vs performance computation."""
        np.random.seed(42)
        n = 200
        
        # Create synthetic data
        y_true = np.random.binomial(1, 0.3, n)
        mean_prob = np.random.beta(2, 3, n)  # Somewhat calibrated
        std_prob = np.random.exponential(0.1, n)
        
        results = compute_coverage_vs_performance(y_true, mean_prob, std_prob)
        
        assert "std_thresholds" in results
        assert "coverage" in results
        assert "accuracy" in results
        assert "auroc" in results
        
        # Check shapes
        assert len(results["coverage"]) == len(results["std_thresholds"])
        assert len(results["accuracy"]) == len(results["std_thresholds"])
        
        # Coverage should be decreasing (more abstention with higher threshold)
        coverage = results["coverage"]
        assert coverage[0] >= coverage[-1]  # Early values >= later values
        
        # Coverage should be between 0 and 1
        assert np.all((coverage >= 0) & (coverage <= 1))


class TestUncertaintyHistogramData:
    """Test uncertainty histogram data preparation."""
    
    def test_uncertainty_histogram_data(self):
        """Test histogram data preparation."""
        np.random.seed(42)
        std_prob = np.random.exponential(0.1, 1000)
        entropy = np.random.exponential(0.2, 1000)
        
        hist_data = uncertainty_histogram_data(std_prob, entropy, n_bins=20)
        
        assert "std_hist" in hist_data
        assert "entropy_hist" in hist_data
        
        # Each histogram should return (counts, bin_edges)
        std_counts, std_bins = hist_data["std_hist"]
        ent_counts, ent_bins = hist_data["entropy_hist"]
        
        assert len(std_counts) == 20
        assert len(std_bins) == 21  # n_bins + 1
        assert len(ent_counts) == 20
        assert len(ent_bins) == 21
        
        # Counts should sum to total samples
        assert np.sum(std_counts) == 1000
        assert np.sum(ent_counts) == 1000


class TestIntegration:
    """Integration tests for uncertainty pipeline."""
    
    def test_end_to_end_uncertainty(self, toy_model, toy_data):
        """Test complete uncertainty quantification pipeline."""
        # MC-Dropout prediction
        mc_results = mc_predict(toy_model, toy_data, T=10)
        
        # Apply abstention
        abstention_results = apply_abstention(
            mc_results["mean_prob"],
            mc_results["std_prob"],
            std_cut=0.1
        )
        
        # Check consistency
        n_samples = len(mc_results["mean_prob"])
        assert len(abstention_results["predictions"]) == n_samples
        assert len(abstention_results["abstained_mask"]) == n_samples
        
        # Abstained samples should have -1 predictions
        abstained_preds = abstention_results["predictions"][abstention_results["abstained_mask"]]
        if len(abstained_preds) > 0:
            assert np.all(abstained_preds == -1)
        
        # Confident samples should have 0 or 1 predictions
        confident_preds = abstention_results["predictions"][abstention_results["confident_mask"]]
        if len(confident_preds) > 0:
            assert np.all((confident_preds == 0) | (confident_preds == 1))
    
    def test_reproducibility(self, toy_model, toy_data):
        """Test that results are reproducible with same seed."""
        torch.manual_seed(123)
        results1 = mc_predict(toy_model, toy_data, T=5)
        
        torch.manual_seed(123)
        results2 = mc_predict(toy_model, toy_data, T=5)
        
        # Results should be identical with same seed
        assert np.allclose(results1["mean_prob"], results2["mean_prob"])
        assert np.allclose(results1["std_prob"], results2["std_prob"])
        assert np.allclose(results1["probs_T"], results2["probs_T"])