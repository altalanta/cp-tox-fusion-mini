"""Feature extraction and processing for multimodal data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings


def create_synthetic_features(n_samples: int = 100, 
                             n_features: int = 50,
                             n_plates: int = 3,
                             random_state: int = 42) -> pd.DataFrame:
    """Create synthetic feature data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of feature columns
        n_plates: Number of plates
        random_state: Random seed
        
    Returns:
        DataFrame with synthetic features and metadata
    """
    np.random.seed(random_state)
    
    # Generate features
    features = np.random.randn(n_samples, n_features)
    
    # Add some structure to make it more realistic
    # Some features correlated within groups
    for i in range(0, n_features, 5):
        end_idx = min(i + 5, n_features)
        group_effect = np.random.randn(n_samples, 1)
        features[:, i:end_idx] += 0.3 * group_effect
    
    # Create feature names
    feature_names = [f"feature_{i:03d}" for i in range(n_features)]
    
    # Create metadata
    compound_ids = [f"COMPOUND_{i%20:03d}" for i in range(n_samples)]
    plate_ids = [f"plate_{(i//20)%n_plates + 1:03d}" for i in range(n_samples)]
    well_rows = [chr(65 + (i//12)%8) for i in range(n_samples)]  # A-H
    well_cols = [str((i%12) + 1) for i in range(n_samples)]      # 1-12
    
    # Create target with some signal
    # Make some compounds more likely to be active
    compound_effects = {f"COMPOUND_{i:03d}": np.random.randn() for i in range(20)}
    targets = []
    for compound_id in compound_ids:
        base_prob = 1 / (1 + np.exp(-compound_effects[compound_id]))
        targets.append(np.random.binomial(1, base_prob))
    
    # Combine into DataFrame
    data = pd.DataFrame(features, columns=feature_names)
    data['compound_id'] = compound_ids
    data['plate_id'] = plate_ids
    data['well_row'] = well_rows
    data['well_col'] = well_cols
    data['target'] = targets
    
    return data


def process_features(data: pd.DataFrame,
                    feature_cols: Optional[List[str]] = None,
                    target_col: str = "target",
                    remove_low_variance: bool = True,
                    variance_threshold: float = 0.01) -> Tuple[pd.DataFrame, Dict]:
    """Process and clean feature data.
    
    Args:
        data: Input DataFrame
        feature_cols: List of feature column names (auto-detect if None)
        target_col: Target column name
        remove_low_variance: Whether to remove low-variance features
        variance_threshold: Minimum variance threshold
        
    Returns:
        Tuple of (processed_data, processing_info)
    """
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [target_col, "compound_id", "plate_id", "well_row", "well_col"]
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    processing_info = {
        "original_features": len(feature_cols),
        "original_samples": len(data),
        "steps_applied": []
    }
    
    # Start with copy of data
    processed_data = data.copy()
    
    # Remove features with too many missing values
    missing_threshold = 0.5
    features_to_keep = []
    for col in feature_cols:
        missing_frac = processed_data[col].isnull().mean()
        if missing_frac <= missing_threshold:
            features_to_keep.append(col)
    
    if len(features_to_keep) < len(feature_cols):
        removed_count = len(feature_cols) - len(features_to_keep)
        processing_info["steps_applied"].append(f"Removed {removed_count} features with >{missing_threshold*100}% missing values")
        feature_cols = features_to_keep
    
    # Fill remaining missing values with median
    for col in feature_cols:
        if processed_data[col].isnull().any():
            median_val = processed_data[col].median()
            processed_data[col].fillna(median_val, inplace=True)
    
    processing_info["steps_applied"].append("Filled missing values with median")
    
    # Remove low-variance features
    if remove_low_variance:
        variances = processed_data[feature_cols].var()
        high_var_features = variances[variances >= variance_threshold].index.tolist()
        
        if len(high_var_features) < len(feature_cols):
            removed_count = len(feature_cols) - len(high_var_features)
            processing_info["steps_applied"].append(f"Removed {removed_count} low-variance features (var < {variance_threshold})")
            feature_cols = high_var_features
    
    # Remove samples with missing target
    if target_col in processed_data.columns:
        initial_samples = len(processed_data)
        processed_data = processed_data[processed_data[target_col].notna()]
        removed_samples = initial_samples - len(processed_data)
        if removed_samples > 0:
            processing_info["steps_applied"].append(f"Removed {removed_samples} samples with missing target")
    
    processing_info.update({
        "final_features": len(feature_cols),
        "final_samples": len(processed_data),
        "feature_columns": feature_cols
    })
    
    return processed_data, processing_info


def fuse_modalities(cp_data: pd.DataFrame, 
                   chem_data: pd.DataFrame,
                   join_col: str = "compound_id",
                   cp_prefix: str = "cp_",
                   chem_prefix: str = "chem_") -> pd.DataFrame:
    """Fuse Cell Painting and chemical features.
    
    Args:
        cp_data: Cell Painting feature data
        chem_data: Chemical feature data  
        join_col: Column to join on
        cp_prefix: Prefix for CP features
        chem_prefix: Prefix for chemical features
        
    Returns:
        Fused DataFrame
    """
    # Prepare CP data
    cp_feature_cols = cp_data.select_dtypes(include=[np.number]).columns.tolist()
    cp_meta_cols = [col for col in cp_data.columns if col not in cp_feature_cols or col == join_col]
    
    cp_renamed = cp_data.copy()
    for col in cp_feature_cols:
        if col != join_col:
            cp_renamed = cp_renamed.rename(columns={col: f"{cp_prefix}{col}"})
    
    # Prepare chemical data
    chem_feature_cols = chem_data.select_dtypes(include=[np.number]).columns.tolist()
    chem_renamed = chem_data.copy()
    for col in chem_feature_cols:
        if col != join_col:
            chem_renamed = chem_renamed.rename(columns={col: f"{chem_prefix}{col}"})
    
    # Fuse on compound ID
    fused_data = pd.merge(cp_renamed, chem_renamed, on=join_col, how="inner")
    
    print(f"Fused {len(cp_data)} CP samples with {len(chem_data)} chemical samples")
    print(f"Result: {len(fused_data)} fused samples")
    
    return fused_data


if __name__ == "__main__":
    # CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python features.py <command> [args...]")
        print("Commands:")
        print("  synthetic [n_samples] [output.parquet] - Generate synthetic features")
        print("  process <input.parquet> [output.parquet] - Process feature data")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "synthetic":
        n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        output_path = sys.argv[3] if len(sys.argv) > 3 else "data/processed/synthetic_features.parquet"
        
        # Generate synthetic data
        data = create_synthetic_features(n_samples=n_samples)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(output_path, index=False)
        
        print(f"Generated {len(data)} samples with {len(data.columns)} columns")
        print(f"Saved to {output_path}")
        
    elif command == "process":
        input_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "data/processed/processed_features.parquet"
        
        # Load and process
        data = pd.read_parquet(input_path)
        processed_data, info = process_features(data)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        processed_data.to_parquet(output_path, index=False)
        
        print(f"Processing completed:")
        for step in info["steps_applied"]:
            print(f"  - {step}")
        print(f"Final: {info['final_samples']} samples, {info['final_features']} features")
        print(f"Saved to {output_path}")
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)