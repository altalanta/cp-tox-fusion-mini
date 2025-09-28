"""Tests for IC50 dose-response modeling."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from cp_tox_mini.dose_response import (
    hill_equation,
    estimate_ic50,
    create_synthetic_dose_response_data,
    process_dose_response_data
)


class TestHillEquation:
    """Test Hill equation implementation."""
    
    def test_hill_equation_basic(self):
        """Test basic Hill equation calculation."""
        doses = np.array([0.1, 1.0, 10.0])
        top, bottom, ic50, hill_slope = 100.0, 0.0, 1.0, 1.0
        
        responses = hill_equation(doses, top, bottom, ic50, hill_slope)
        
        # Check basic properties
        assert len(responses) == len(doses)
        assert all(np.isfinite(responses))
        assert responses[1] == pytest.approx((top + bottom) / 2, rel=0.01)  # At IC50
        
    def test_hill_equation_edge_cases(self):
        """Test Hill equation with edge cases."""
        # Zero doses
        doses = np.array([0.0, 1e-12, 1.0])
        responses = hill_equation(doses, 100, 0, 1.0, 1.0)
        assert all(np.isfinite(responses))
        
        # Very large doses
        doses = np.array([1e-6, 1.0, 1e6])
        responses = hill_equation(doses, 100, 0, 1.0, 1.0)
        assert all(np.isfinite(responses))


class TestIC50Estimation:
    """Test IC50 estimation functionality."""
    
    def test_perfect_synthetic_data(self):
        """Test IC50 estimation on perfect synthetic data."""
        # Generate perfect Hill curve
        doses = np.logspace(-2, 2, 20)  # 0.01 to 100
        true_ic50 = 1.0
        true_top = 100.0
        true_bottom = 0.0
        true_hill_slope = 1.0
        
        responses = hill_equation(doses, true_top, true_bottom, true_ic50, true_hill_slope)
        
        # Estimate IC50
        result = estimate_ic50(doses, responses)
        
        # Check result
        assert result["fit_success"] is True
        assert result["ic50"] == pytest.approx(true_ic50, rel=0.1)
        assert result["top"] == pytest.approx(true_top, rel=0.1)
        assert result["bottom"] == pytest.approx(true_bottom, abs=5.0)
        assert result["r_squared"] > 0.95
        
    def test_noisy_synthetic_data(self):
        """Test IC50 estimation with noisy data."""
        doses = np.logspace(-1, 1, 10)
        true_ic50 = 1.0
        
        # Add noise
        np.random.seed(42)
        responses = hill_equation(doses, 100, 0, true_ic50, 1.0)
        noisy_responses = responses + np.random.normal(0, 5, len(responses))
        
        result = estimate_ic50(doses, noisy_responses)
        
        assert result["fit_success"] is True
        assert result["ic50"] == pytest.approx(true_ic50, rel=0.5)  # More tolerance for noise
        assert result["r_squared"] > 0.7  # Still reasonable fit
        
    def test_insufficient_data(self):
        """Test IC50 estimation with insufficient data."""
        doses = np.array([0.1, 1.0])  # Only 2 points
        responses = np.array([90, 10])
        
        result = estimate_ic50(doses, responses)
        
        assert result["fit_success"] is False
        assert "Insufficient data points" in result["error"]
        
    def test_invalid_data(self):
        """Test IC50 estimation with invalid data."""
        # All NaN responses
        doses = np.array([0.1, 1.0, 10.0, 100.0])
        responses = np.array([np.nan, np.nan, np.nan, np.nan])
        
        result = estimate_ic50(doses, responses)
        
        assert result["fit_success"] is False
        assert "Insufficient valid data points" in result["error"]
        
    def test_known_ic50_validation(self):
        """Test validation against known IC50 values."""
        test_cases = [
            {"ic50": 0.1, "tolerance": 0.2},
            {"ic50": 1.0, "tolerance": 0.2},
            {"ic50": 10.0, "tolerance": 0.2},
        ]
        
        for case in test_cases:
            doses = np.logspace(-3, 3, 15)
            true_ic50 = case["ic50"]
            responses = hill_equation(doses, 100, 0, true_ic50, 1.0)
            
            result = estimate_ic50(doses, responses)
            
            assert result["fit_success"] is True
            relative_error = abs(result["ic50"] - true_ic50) / true_ic50
            assert relative_error < case["tolerance"], f"IC50 {true_ic50}: error {relative_error:.3f} > {case['tolerance']}"


class TestSyntheticDataGeneration:
    """Test synthetic dose-response data generation."""
    
    def test_synthetic_data_structure(self):
        """Test structure of synthetic dose-response data."""
        data = create_synthetic_dose_response_data(n_compounds=3, n_points_per_compound=8, random_state=42)
        
        # Check structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3 * 8  # 3 compounds × 8 points
        
        required_columns = ["compound_id", "dose", "response", "true_ic50"]
        for col in required_columns:
            assert col in data.columns
            
        # Check compound IDs
        compounds = data["compound_id"].unique()
        assert len(compounds) == 3
        
        # Check dose range
        assert data["dose"].min() > 0
        assert data["dose"].max() > data["dose"].min()
        
    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data generation is reproducible."""
        data1 = create_synthetic_dose_response_data(n_compounds=2, random_state=42)
        data2 = create_synthetic_dose_response_data(n_compounds=2, random_state=42)
        
        pd.testing.assert_frame_equal(data1, data2)
        
    def test_synthetic_data_ic50_recovery(self):
        """Test that we can recover known IC50s from synthetic data."""
        data = create_synthetic_dose_response_data(n_compounds=2, n_points_per_compound=10, random_state=42)
        
        for compound_id in data["compound_id"].unique():
            compound_data = data[data["compound_id"] == compound_id]
            
            doses = compound_data["dose"].values
            responses = compound_data["response"].values
            true_ic50 = compound_data["true_ic50"].iloc[0]
            
            result = estimate_ic50(doses, responses)
            
            if result["fit_success"]:
                relative_error = abs(result["ic50"] - true_ic50) / true_ic50
                # Allow for some noise in synthetic data
                assert relative_error < 0.3, f"Compound {compound_id}: error {relative_error:.3f} too high"


class TestProcessDoseResponseData:
    """Test dose-response data processing pipeline."""
    
    def test_process_synthetic_data(self, tmp_path):
        """Test processing of synthetic dose-response data."""
        # Generate synthetic data
        data = create_synthetic_dose_response_data(n_compounds=3, n_points_per_compound=8, random_state=42)
        
        # Process data
        results = process_dose_response_data(
            data, 
            output_dir=tmp_path,
            plot_curves=False  # Skip plotting for speed
        )
        
        # Check results
        assert isinstance(results, dict)
        assert len(results) <= 3  # At most 3 compounds
        
        # Check that JSON output was created
        json_file = tmp_path / "ic50_summary.json"
        assert json_file.exists()
        
        # Validate at least some fits were successful
        successful_fits = sum(1 for r in results.values() if r.get("fit_success", False))
        assert successful_fits > 0, "No successful IC50 fits"
        
    def test_process_empty_data(self, tmp_path):
        """Test processing with empty data."""
        empty_data = pd.DataFrame(columns=["compound_id", "dose", "response"])
        
        results = process_dose_response_data(empty_data, output_dir=tmp_path)
        
        assert isinstance(results, dict)
        assert len(results) == 0
        
    def test_process_insufficient_data_per_compound(self, tmp_path):
        """Test processing with insufficient data per compound."""
        # Only 2 points per compound (need ≥4)
        data = pd.DataFrame({
            "compound_id": ["A", "A", "B", "B"],
            "dose": [0.1, 1.0, 0.1, 1.0],
            "response": [90, 50, 95, 60]
        })
        
        results = process_dose_response_data(data, output_dir=tmp_path)
        
        # Should skip compounds with insufficient data
        assert len(results) == 0


@pytest.fixture
def sample_dose_response_data():
    """Fixture providing sample dose-response data."""
    return create_synthetic_dose_response_data(n_compounds=2, n_points_per_compound=8, random_state=42)


def test_ic50_estimation_integration(sample_dose_response_data, tmp_path):
    """Integration test for complete IC50 workflow."""
    # Process the sample data
    results = process_dose_response_data(
        sample_dose_response_data,
        output_dir=tmp_path,
        plot_curves=True
    )
    
    # Check that processing completed
    assert len(results) > 0
    
    # Check that output files were created
    assert (tmp_path / "ic50_summary.json").exists()
    
    # Validate that at least one compound had successful fit
    successful_compounds = [
        cid for cid, result in results.items() 
        if result.get("fit_success", False)
    ]
    assert len(successful_compounds) > 0, "No compounds had successful IC50 fits"
    
    # For successful fits, check that IC50 is reasonable
    for compound_id in successful_compounds:
        result = results[compound_id]
        ic50 = result["ic50"]
        
        # IC50 should be positive and within reasonable range
        assert ic50 > 0
        assert ic50 < 1000  # Reasonable upper bound for test data
        
        # R² should indicate decent fit
        assert result["r_squared"] > 0.5