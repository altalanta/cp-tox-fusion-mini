"""Leakage and batch effect diagnostics for data quality assessment."""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def plate_layout_probe(
    features: pd.DataFrame,
    plate_id_col: str = "plate_id",
    well_row_col: str = "well_row", 
    well_col_col: str = "well_col",
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, float]:
    """Train models to predict plate/layout info from features.
    
    High performance indicates potential plate or layout confounding.
    
    Args:
        features: DataFrame with features and metadata
        plate_id_col: Column name for plate identifier
        well_row_col: Column name for well row (A, B, C, ...)
        well_col_col: Column name for well column (1, 2, 3, ...)
        feature_cols: List of feature columns to use (if None, use all numeric)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with probe metrics including max_score
    """
    # Select feature columns
    if feature_cols is None:
        feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        # Remove metadata columns
        exclude_cols = [plate_id_col, well_row_col, well_col_col, "target", "compound_id"]
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        return {"error": "No numeric feature columns found"}
    
    # Prepare features
    X = features[feature_cols].fillna(0)  # Simple imputation
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # Plate ID probe
    if plate_id_col in features.columns:
        plate_ids = features[plate_id_col].dropna()
        if len(plate_ids.unique()) > 1:
            # Encode plate IDs
            le_plate = LabelEncoder()
            y_plate = le_plate.fit_transform(plate_ids)
            
            # Match X to non-null plate IDs
            valid_idx = features[plate_id_col].notna()
            X_plate = X_scaled[valid_idx]
            
            try:
                # Train simple classifier
                clf_plate = LogisticRegression(random_state=random_state, max_iter=1000)
                
                if len(np.unique(y_plate)) > 2:
                    # Multi-class
                    scores = cross_val_score(clf_plate, X_plate, y_plate, 
                                           cv=min(3, len(np.unique(y_plate))), 
                                           scoring='roc_auc_ovr')
                else:
                    # Binary
                    scores = cross_val_score(clf_plate, X_plate, y_plate, 
                                           cv=3, scoring='roc_auc')
                
                results["plate_id_score"] = float(np.mean(scores))
                
            except Exception as e:
                results["plate_probe_auroc"] = 0.0
                results["plate_probe_error"] = str(e)
    
    # Well row probe  
    if well_row_col in features.columns:
        well_rows = features[well_row_col].dropna()
        if len(well_rows.unique()) > 1:
            le_row = LabelEncoder()
            y_row = le_row.fit_transform(well_rows)
            
            valid_idx = features[well_row_col].notna()
            X_row = X_scaled[valid_idx]
            
            try:
                clf_row = LogisticRegression(random_state=random_state, max_iter=1000)
                
                if len(np.unique(y_row)) > 2:
                    scores = cross_val_score(clf_row, X_row, y_row, 
                                           cv=min(3, len(np.unique(y_row))), 
                                           scoring='roc_auc_ovr')
                else:
                    scores = cross_val_score(clf_row, X_row, y_row, 
                                           cv=3, scoring='roc_auc')
                
                results["well_row_score"] = float(np.mean(scores))
                
            except Exception as e:
                results["well_row_probe_auroc"] = 0.0
                results["well_row_probe_error"] = str(e)
    
    # Well column probe
    if well_col_col in features.columns:
        well_cols = features[well_col_col].dropna()
        if len(well_cols.unique()) > 1:
            le_col = LabelEncoder()
            y_col = le_col.fit_transform(well_cols)
            
            valid_idx = features[well_col_col].notna()
            X_col = X_scaled[valid_idx]
            
            try:
                clf_col = LogisticRegression(random_state=random_state, max_iter=1000)
                
                if len(np.unique(y_col)) > 2:
                    scores = cross_val_score(clf_col, X_col, y_col, 
                                           cv=min(3, len(np.unique(y_col))), 
                                           scoring='roc_auc_ovr')
                else:
                    scores = cross_val_score(clf_col, X_col, y_col, 
                                           cv=3, scoring='roc_auc')
                
                results["well_col_score"] = float(np.mean(scores))
                
            except Exception as e:
                results["well_col_probe_auroc"] = 0.0
                results["well_col_probe_error"] = str(e)
    
    # Combined layout probe (row + col)
    if well_row_col in features.columns and well_col_col in features.columns:
        # Create combined well position identifier
        well_positions = features[well_row_col].astype(str) + "_" + features[well_col_col].astype(str)
        well_positions = well_positions.dropna()
        
        if len(well_positions.unique()) > 1:
            le_pos = LabelEncoder()
            y_pos = le_pos.fit_transform(well_positions)
            
            valid_idx = (features[well_row_col].notna() & features[well_col_col].notna())
            X_pos = X_scaled[valid_idx]
            
            try:
                clf_pos = LogisticRegression(random_state=random_state, max_iter=1000)
                
                if len(np.unique(y_pos)) > 2:
                    scores = cross_val_score(clf_pos, X_pos, y_pos, 
                                           cv=min(3, len(np.unique(y_pos))), 
                                           scoring='roc_auc_ovr')
                else:
                    scores = cross_val_score(clf_pos, X_pos, y_pos, 
                                           cv=3, scoring='roc_auc')
                
                results["layout_probe_auroc"] = float(np.mean(scores))
                
            except Exception as e:
                results["layout_probe_auroc"] = 0.0
                results["layout_probe_error"] = str(e)
    
    # Calculate max score across all probes
    score_keys = ["plate_id_score", "well_row_score", "well_col_score", "layout_probe_auroc"]
    scores = [results.get(key, 0.0) for key in score_keys if key in results]
    results["max_score"] = max(scores) if scores else 0.0
    
    return results


def plate_effect_on_target(
    features: pd.DataFrame,
    target: Union[str, np.ndarray, List] = "target",
    plate_id_col: str = "plate_id",
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, float]:
    """Assess effect of including plate dummies on target prediction.
    
    Large improvement suggests plate confounding.
    
    Args:
        features: DataFrame with features and plate info
        target: Target variable (column name or array-like)
        plate_id_col: Column name for plate identifier
        feature_cols: List of feature columns to use
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with baseline and plate-adjusted performance
    """
    # Handle target input
    if isinstance(target, str):
        target_col = target
        if target_col not in features.columns:
            return {"error": f"Target column '{target_col}' not found"}
        y = features[target_col]
    else:
        y = np.array(target)
        if len(y) != len(features):
            return {"error": "Target array length doesn't match features"}
    
    if plate_id_col not in features.columns:
        return {"error": f"Plate column '{plate_id_col}' not found"}
    
    # Select feature columns
    if feature_cols is None:
        feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [plate_id_col, "well_row", "well_col", "compound_id"]
        if isinstance(target, str):
            exclude_cols.append(target)
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        return {"error": "No numeric feature columns found"}
    
    # Prepare data - handle missing values
    plate_valid = features[plate_id_col].notna()
    if isinstance(target, str):
        target_valid = features[target].notna()
        valid_idx = plate_valid & target_valid
    else:
        y_series = pd.Series(y)
        target_valid = y_series.notna()
        valid_idx = plate_valid & target_valid
    
    data_clean = features[valid_idx].copy()
    y_clean = y[valid_idx] if not isinstance(target, str) else data_clean[target]
    
    if len(data_clean) < 10:
        return {"error": "Insufficient data for analysis"}
    
    X_base = data_clean[feature_cols].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)
    
    results = {}
    
    try:
        # Baseline model (no plate dummies)
        clf_base = RandomForestClassifier(
            n_estimators=50, random_state=random_state, max_depth=5
        )
        
        if len(np.unique(y_clean)) > 2:
            scores_base = cross_val_score(clf_base, X_base_scaled, y_clean, 
                                        cv=3, scoring='roc_auc_ovr')
        else:
            scores_base = cross_val_score(clf_base, X_base_scaled, y_clean, 
                                        cv=3, scoring='roc_auc')
        
        baseline_auroc = float(np.mean(scores_base))
        results["baseline_target_auroc"] = baseline_auroc
        
        # Model with plate dummies
        plate_dummies = pd.get_dummies(data_clean[plate_id_col], prefix='plate')
        X_with_plates = np.concatenate([X_base_scaled, plate_dummies.values], axis=1)
        
        clf_plates = RandomForestClassifier(
            n_estimators=50, random_state=random_state, max_depth=5
        )
        
        if len(np.unique(y_clean)) > 2:
            scores_plates = cross_val_score(clf_plates, X_with_plates, y_clean, 
                                          cv=3, scoring='roc_auc_ovr')
        else:
            scores_plates = cross_val_score(clf_plates, X_with_plates, y_clean, 
                                          cv=3, scoring='roc_auc')
        
        plate_adjusted_auroc = float(np.mean(scores_plates))
        results["plate_adjusted_target_auroc"] = plate_adjusted_auroc
        
        # Compute delta and standardized names
        results["plate_effect_score"] = plate_adjusted_auroc - baseline_auroc
        results["cross_val_score"] = plate_adjusted_auroc  
        results["baseline_score"] = baseline_auroc
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


def permutation_test(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[str, np.ndarray, List],
    model,
    n_permutations: int = 100,
    random_state: int = 42
) -> Dict[str, float]:
    """Permutation test for overfitting detection.
    
    Args:
        X: Feature matrix (DataFrame or array)
        y: Target vector (array-like) or column name if X is DataFrame
        model: sklearn-style model with fit() and score() methods
        n_permutations: Number of permutation iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with permutation test results
    """
    np.random.seed(random_state)
    
    # Handle input types
    if isinstance(X, pd.DataFrame):
        if isinstance(y, str):
            # y is column name
            if y not in X.columns:
                return {"error": f"Target column '{y}' not found"}
            y_array = X[y].values
            X_array = X.drop(columns=[y]).values
        else:
            # y is array-like
            y_array = np.array(y)
            X_array = X.values
    else:
        # X is array-like
        X_array = np.array(X)
        y_array = np.array(y)
    
    if len(X_array) != len(y_array):
        return {"error": "X and y must have same length"}
    
    if len(X_array) < 10:
        return {"error": "Insufficient data for permutation test"}
    
    # Original model performance
    try:
        model.fit(X_array, y_array)
        original_score = model.score(X_array, y_array)
    except Exception as e:
        return {"error": f"Failed to fit model: {e}"}
    
    # Permutation tests
    permuted_scores = []
    
    for i in range(n_permutations):
        # Permute target labels
        y_permuted = y_array.copy()
        np.random.shuffle(y_permuted)
        
        try:
            # Fit model on permuted data
            model.fit(X_array, y_permuted)
            perm_score = model.score(X_array, y_permuted)
            permuted_scores.append(perm_score)
        except Exception:
            # Skip failed permutations
            continue
    
    if len(permuted_scores) == 0:
        return {"error": "All permutations failed"}
    
    permuted_scores = np.array(permuted_scores)
    
    # Compute p-value (one-tailed test: original score > permuted scores)
    p_value = float(np.mean(permuted_scores >= original_score))
    
    results = {
        "original_score": float(original_score),
        "permuted_scores": permuted_scores.tolist(),
        "p_value": p_value,
        "is_significant": p_value <= 0.05,
        "n_permutations_completed": len(permuted_scores)
    }
    
    return results


def assess_leakage_risk(diagnostics: Dict) -> Dict[str, Union[str, List[str]]]:
    """Assess overall leakage risk from diagnostic results.
    
    Args:
        diagnostics: Dictionary with diagnostic metrics
        
    Returns:
        Dictionary with overall risk and specific risk factors
    """
    risk_factors = []
    
    # Extract scores from nested diagnostic results
    plate_layout = diagnostics.get("plate_layout_probe", {})
    plate_effect = diagnostics.get("plate_effect_on_target", {})
    permutation = diagnostics.get("permutation_test", {})
    
    max_probe_score = plate_layout.get("max_score", 0)
    plate_effect_score = plate_effect.get("plate_effect_score", 0) 
    perm_p_value = permutation.get("p_value", 1)
    
    # Check high risk conditions
    if max_probe_score >= 0.8:
        risk_factors.append("High plate/layout predictability")
    
    if plate_effect_score >= 0.02:
        risk_factors.append("Strong plate effect on target")
    
    if perm_p_value <= 0.05:
        risk_factors.append("Significant overfitting detected")
    
    # Determine overall risk
    if len(risk_factors) >= 2 or max_probe_score >= 0.9:
        overall_risk = "high"
    elif len(risk_factors) >= 1 or max_probe_score >= 0.7:
        overall_risk = "medium"  
    else:
        overall_risk = "low"
    
    return {
        "overall_risk": overall_risk,
        "risk_factors": risk_factors
    }


def run_full_diagnostics(
    features: pd.DataFrame,
    target: Union[str, np.ndarray, List] = "target",
    plate_id_col: str = "plate_id", 
    well_row_col: str = "well_row",
    well_col_col: str = "well_col",
    feature_cols: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    random_state: int = 42
) -> Dict:
    """Run complete leakage/batch diagnostics.
    
    Args:
        features: DataFrame with features and metadata
        target_col: Column name for target variable
        plate_id_col: Column name for plate identifier
        well_row_col: Column name for well row
        well_col_col: Column name for well column
        feature_cols: List of feature columns to use
        output_path: Path to save results JSON
        random_state: Random seed for reproducibility
        
    Returns:
        Complete diagnostics dictionary
    """
    print("Running leakage/batch diagnostics...")
    
    results = {}
    
    # Plate/layout probes
    print("  Running plate/layout probes...")
    probe_results = plate_layout_probe(
        features, plate_id_col, well_row_col, well_col_col, 
        feature_cols, random_state
    )
    results["plate_layout_probe"] = probe_results
    
    # Plate effect on target
    print("  Assessing plate effect on target...")
    plate_effect = plate_effect_on_target(
        features, target, plate_id_col, feature_cols, random_state
    )
    results["plate_effect_on_target"] = plate_effect
    
    # Permutation test with a simple model
    print("  Running permutation test...")
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # Prepare data for permutation test
    if isinstance(target, str):
        if target not in features.columns:
            print(f"    Warning: Target column '{target}' not found, skipping permutation test")
            perm_results = {"error": f"Target column '{target}' not found"}
        else:
            # Get clean data for permutation test
            valid_idx = features[target].notna()
            clean_features = features[valid_idx]
            
            if feature_cols is None:
                feature_cols_perm = clean_features.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = [target, plate_id_col, well_row_col, well_col_col, "compound_id"]
                feature_cols_perm = [col for col in feature_cols_perm if col not in exclude_cols]
            else:
                feature_cols_perm = feature_cols
                
            if len(feature_cols_perm) > 0:
                X_perm = clean_features[feature_cols_perm].fillna(0)
                y_perm = clean_features[target]
                perm_results = permutation_test(X_perm, y_perm, model, n_permutations=50, random_state=random_state)
            else:
                perm_results = {"error": "No feature columns available for permutation test"}
    else:
        # Target is array-like
        if feature_cols is None:
            feature_cols_perm = features.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = [plate_id_col, well_row_col, well_col_col, "compound_id"]
            feature_cols_perm = [col for col in feature_cols_perm if col not in exclude_cols]
        else:
            feature_cols_perm = feature_cols
            
        if len(feature_cols_perm) > 0:
            X_perm = features[feature_cols_perm].fillna(0)
            perm_results = permutation_test(X_perm, target, model, n_permutations=50, random_state=random_state)
        else:
            perm_results = {"error": "No feature columns available for permutation test"}
    
    results["permutation_test"] = perm_results
    
    # Assess overall risk
    print("  Assessing overall leakage risk...")
    risk_assessment = assess_leakage_risk(results)
    results["leakage_risk"] = risk_assessment
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"  Saved diagnostics to {output_path}")
    
    print(f"  Risk assessment: {risk_assessment['overall_risk'].upper()}")
    
    return results


if __name__ == "__main__":
    # CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnostics.py <data_file.csv> [output.json]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "diagnostics_results.json"
    
    try:
        df = pd.read_csv(data_file)
        results = run_full_diagnostics(df, output_path=output_file)
        
        print("\nDiagnostics Summary:")
        print(f"Risk Level: {results.get('risk_of_leakage', 'unknown')}")
        print(f"Notes: {results.get('notes', 'none')}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)