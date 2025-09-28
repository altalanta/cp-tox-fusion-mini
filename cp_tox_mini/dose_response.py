"""Dose-response modeling and IC50 estimation."""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def hill_equation(dose: np.ndarray, top: float, bottom: float, ic50: float, hill_slope: float) -> np.ndarray:
    """Four-parameter log-logistic (Hill) equation for dose-response curves.
    
    Args:
        dose: Dose/concentration values
        top: Maximum response (upper asymptote)
        bottom: Minimum response (lower asymptote)  
        ic50: Half-maximal inhibitory concentration
        hill_slope: Hill slope coefficient (steepness)
        
    Returns:
        Predicted response values
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Handle zero and negative doses
        dose = np.maximum(dose, 1e-12)
        
        # Four-parameter log-logistic model
        response = bottom + (top - bottom) / (1 + (dose / ic50) ** hill_slope)
        
        # Handle numerical overflow/underflow
        response = np.nan_to_num(response, nan=bottom, posinf=top, neginf=bottom)
        
        return response


def estimate_ic50(doses: np.ndarray, responses: np.ndarray,
                  bounds: Tuple[List, List] = None,
                  initial_guess: List[float] = None,
                  max_dose: Optional[float] = None) -> Dict[str, Union[float, bool, str]]:
    """Estimate IC50 from dose-response data using 4-parameter log-logistic model.
    
    Args:
        doses: Array of dose/concentration values
        responses: Array of response values (e.g., % viability, % inhibition)
        bounds: Parameter bounds as (lower_bounds, upper_bounds)
        initial_guess: Initial parameter guess [top, bottom, ic50, hill_slope]
        max_dose: Maximum dose tested (for extrapolation warning)
        
    Returns:
        Dictionary with IC50 estimate and fit statistics
    """
    # Input validation
    doses = np.asarray(doses)
    responses = np.asarray(responses)
    
    if len(doses) != len(responses):
        raise ValueError("Doses and responses must have same length")
    
    if len(doses) < 4:
        return {
            "ic50": np.nan,
            "top": np.nan,
            "bottom": np.nan,
            "hill_slope": np.nan,
            "r_squared": np.nan,
            "fit_success": False,
            "error": "Insufficient data points (need ≥4)"
        }
    
    # Remove invalid data
    valid_idx = np.isfinite(doses) & np.isfinite(responses) & (doses > 0)
    if np.sum(valid_idx) < 4:
        return {
            "ic50": np.nan,
            "top": np.nan, 
            "bottom": np.nan,
            "hill_slope": np.nan,
            "r_squared": np.nan,
            "fit_success": False,
            "error": "Insufficient valid data points"
        }
    
    doses_clean = doses[valid_idx]
    responses_clean = responses[valid_idx]
    
    # Set default bounds if not provided
    if bounds is None:
        response_min, response_max = np.min(responses_clean), np.max(responses_clean)
        response_range = response_max - response_min
        dose_min, dose_max = np.min(doses_clean), np.max(doses_clean)
        
        # Parameter bounds: [top, bottom, ic50, hill_slope]
        lower_bounds = [
            response_min - 0.2 * response_range,  # top
            response_min - 0.2 * response_range,  # bottom  
            dose_min * 0.01,                      # ic50
            0.1                                   # hill_slope
        ]
        upper_bounds = [
            response_max + 0.2 * response_range,  # top
            response_max + 0.2 * response_range,  # bottom
            dose_max * 100,                       # ic50
            10.0                                  # hill_slope
        ]
        bounds = (lower_bounds, upper_bounds)
    
    # Set default initial guess if not provided
    if initial_guess is None:
        response_min, response_max = np.min(responses_clean), np.max(responses_clean)
        dose_mid = np.sqrt(np.min(doses_clean) * np.max(doses_clean))  # Geometric mean
        
        initial_guess = [
            response_max,      # top
            response_min,      # bottom
            dose_mid,          # ic50
            1.0               # hill_slope
        ]
    
    try:
        # Fit curve
        popt, pcov = curve_fit(
            hill_equation, 
            doses_clean, 
            responses_clean,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000
        )
        
        top, bottom, ic50, hill_slope = popt
        
        # Calculate goodness of fit
        y_pred = hill_equation(doses_clean, *popt)
        ss_res = np.sum((responses_clean - y_pred) ** 2)
        ss_tot = np.sum((responses_clean - np.mean(responses_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Parameter uncertainties (standard errors)
        param_errors = np.sqrt(np.diag(pcov))
        
        # Check for extrapolation warning
        extrapolated = False
        if max_dose is not None and ic50 > max_dose:
            extrapolated = True
        
        result = {
            "ic50": float(ic50),
            "top": float(top),
            "bottom": float(bottom), 
            "hill_slope": float(hill_slope),
            "r_squared": float(r_squared),
            "ic50_stderr": float(param_errors[2]),
            "top_stderr": float(param_errors[0]),
            "bottom_stderr": float(param_errors[1]),
            "hill_slope_stderr": float(param_errors[3]),
            "fit_success": True,
            "extrapolated": extrapolated,
            "n_points": len(doses_clean)
        }
        
        return result
        
    except (RuntimeError, ValueError, TypeError) as e:
        return {
            "ic50": np.nan,
            "top": np.nan,
            "bottom": np.nan,
            "hill_slope": np.nan, 
            "r_squared": np.nan,
            "fit_success": False,
            "error": str(e)
        }


def plot_dose_response_curve(doses: np.ndarray, responses: np.ndarray,
                            fit_params: Dict, output_path: Path,
                            title: str = "Dose-Response Curve",
                            xlabel: str = "Dose", ylabel: str = "Response") -> bool:
    """Plot dose-response curve with fitted model.
    
    Args:
        doses: Dose values
        responses: Response values
        fit_params: Parameters from estimate_ic50()
        output_path: Path to save plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        True if successful, False otherwise
    """
    try:
        plt.figure(figsize=(8, 6), dpi=100)
        
        # Remove invalid data for plotting
        valid_idx = np.isfinite(doses) & np.isfinite(responses) & (doses > 0)
        doses_clean = doses[valid_idx]
        responses_clean = responses[valid_idx]
        
        # Plot data points
        plt.scatter(doses_clean, responses_clean, alpha=0.7, s=50, 
                   color='blue', label='Data')
        
        # Plot fitted curve if fit was successful
        if fit_params.get("fit_success", False):
            dose_range = np.logspace(
                np.log10(np.min(doses_clean) * 0.1),
                np.log10(np.max(doses_clean) * 10),
                200
            )
            
            fitted_response = hill_equation(
                dose_range,
                fit_params["top"],
                fit_params["bottom"], 
                fit_params["ic50"],
                fit_params["hill_slope"]
            )
            
            plt.plot(dose_range, fitted_response, 'r-', linewidth=2, 
                    label=f'Fit (IC50 = {fit_params["ic50"]:.2e})')
            
            # Mark IC50 point
            ic50_response = (fit_params["top"] + fit_params["bottom"]) / 2
            plt.axvline(x=fit_params["ic50"], color='red', linestyle='--', alpha=0.7)
            plt.axhline(y=ic50_response, color='red', linestyle='--', alpha=0.7)
            plt.scatter([fit_params["ic50"]], [ic50_response], 
                       color='red', s=100, marker='x', linewidths=3)
        
        plt.xscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² to plot if available
        if "r_squared" in fit_params and not np.isnan(fit_params["r_squared"]):
            plt.text(0.05, 0.95, f'R² = {fit_params["r_squared"]:.3f}',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Failed to plot dose-response curve: {e}")
        return False


def process_dose_response_data(data: pd.DataFrame,
                              dose_col: str = "dose",
                              response_col: str = "response", 
                              compound_col: str = "compound_id",
                              output_dir: Path = Path("reports"),
                              plot_curves: bool = True) -> Dict[str, Dict]:
    """Process dose-response data for multiple compounds.
    
    Args:
        data: DataFrame with dose-response data
        dose_col: Column name for dose values
        response_col: Column name for response values
        compound_col: Column name for compound identifiers
        output_dir: Directory to save results
        plot_curves: Whether to generate individual curve plots
        
    Returns:
        Dictionary of IC50 results by compound
    """
    print(f"Processing dose-response data for {data[compound_col].nunique()} compounds...")
    
    results = {}
    summary_stats = {
        "total_compounds": 0,
        "successful_fits": 0,
        "failed_fits": 0,
        "extrapolated_fits": 0
    }
    
    figures_dir = output_dir / "figures" / "dose_response"
    if plot_curves:
        figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each compound
    for compound_id in data[compound_col].unique():
        compound_data = data[data[compound_col] == compound_id]
        
        if len(compound_data) < 4:
            print(f"  Skipping {compound_id}: insufficient data points ({len(compound_data)} < 4)")
            continue
        
        doses = compound_data[dose_col].values
        responses = compound_data[response_col].values
        
        # Estimate IC50
        ic50_result = estimate_ic50(doses, responses, max_dose=np.max(doses))
        results[compound_id] = ic50_result
        
        # Update summary stats
        summary_stats["total_compounds"] += 1
        if ic50_result["fit_success"]:
            summary_stats["successful_fits"] += 1
            if ic50_result.get("extrapolated", False):
                summary_stats["extrapolated_fits"] += 1
        else:
            summary_stats["failed_fits"] += 1
        
        # Plot individual curve
        if plot_curves and ic50_result["fit_success"]:
            plot_path = figures_dir / f"{compound_id}_dose_response.png"
            plot_dose_response_curve(
                doses, responses, ic50_result, plot_path,
                title=f"Dose-Response Curve: {compound_id}",
                xlabel="Dose (µM)", ylabel="Response (%)"
            )
    
    # Save summary results
    summary_results = {
        "summary_statistics": summary_stats,
        "compound_results": results,
        "metadata": {
            "timestamp_utc": pd.Timestamp.utcnow().isoformat() + "Z",
            "total_data_points": len(data),
            "dose_range": [float(data[dose_col].min()), float(data[dose_col].max())],
            "response_range": [float(data[response_col].min()), float(data[response_col].max())]
        }
    }
    
    # Save to JSON
    output_path = output_dir / "ic50_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(summary_results, f, indent=2, default=str)
    
    print(f"  Processed {summary_stats['total_compounds']} compounds")
    print(f"  Successful fits: {summary_stats['successful_fits']}")
    print(f"  Failed fits: {summary_stats['failed_fits']}")
    print(f"  Extrapolated IC50s: {summary_stats['extrapolated_fits']}")
    print(f"  Results saved to {output_path}")
    
    return results


def create_synthetic_dose_response_data(n_compounds: int = 5, 
                                       n_points_per_compound: int = 8,
                                       noise_level: float = 0.1,
                                       random_state: int = 42) -> pd.DataFrame:
    """Create synthetic dose-response data for testing.
    
    Args:
        n_compounds: Number of compounds to simulate
        n_points_per_compound: Number of dose points per compound
        noise_level: Noise level to add to responses
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic dose-response data
    """
    np.random.seed(random_state)
    
    data = []
    
    for i in range(n_compounds):
        compound_id = f"COMPOUND_{i+1:03d}"
        
        # Random IC50 parameters
        true_top = np.random.uniform(90, 100)
        true_bottom = np.random.uniform(0, 10)
        true_ic50 = np.random.uniform(0.1, 10)  # µM
        true_hill_slope = np.random.uniform(0.5, 3.0)
        
        # Dose range
        doses = np.logspace(-2, 2, n_points_per_compound)  # 0.01 to 100 µM
        
        # True responses
        true_responses = hill_equation(doses, true_top, true_bottom, true_ic50, true_hill_slope)
        
        # Add noise
        noise = np.random.normal(0, noise_level * (true_top - true_bottom), len(doses))
        noisy_responses = true_responses + noise
        
        # Store data
        for dose, response in zip(doses, noisy_responses):
            data.append({
                "compound_id": compound_id,
                "dose": dose,
                "response": response,
                "true_ic50": true_ic50,
                "true_top": true_top,
                "true_bottom": true_bottom,
                "true_hill_slope": true_hill_slope
            })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dose_response.py synthetic [output_dir]  - Generate synthetic data and test")
        print("  python dose_response.py <data.csv> [output_dir] - Process real data")
        sys.exit(1)
    
    command = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports")
    
    if command == "synthetic":
        print("Generating synthetic dose-response data...")
        
        # Create synthetic data
        synthetic_data = create_synthetic_dose_response_data(
            n_compounds=3, n_points_per_compound=8, random_state=42
        )
        
        # Save synthetic data
        synthetic_path = output_dir / "synthetic_dose_response.csv"
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_data.to_csv(synthetic_path, index=False)
        print(f"Saved synthetic data to {synthetic_path}")
        
        # Process synthetic data
        results = process_dose_response_data(
            synthetic_data, output_dir=output_dir, plot_curves=True
        )
        
        # Validate against known values
        print("\nValidation against known IC50 values:")
        for compound_id, result in results.items():
            if result["fit_success"]:
                true_ic50 = synthetic_data[synthetic_data["compound_id"] == compound_id]["true_ic50"].iloc[0]
                estimated_ic50 = result["ic50"]
                relative_error = abs(estimated_ic50 - true_ic50) / true_ic50 * 100
                print(f"  {compound_id}: True={true_ic50:.3f}, Est={estimated_ic50:.3f}, Error={relative_error:.1f}%")
        
    else:
        # Process real data file
        data_file = command
        
        try:
            data = pd.read_csv(data_file)
            results = process_dose_response_data(data, output_dir=output_dir, plot_curves=True)
            print("Processing completed successfully!")
            
        except Exception as e:
            print(f"Error processing {data_file}: {e}")
            sys.exit(1)