"""Model evaluation and metrics computation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Confidence in this bin
            confidence_in_bin = y_prob[in_bin].mean()
            # Accuracy in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            # Add to ECE
            ece += np.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for probabilistic metrics)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic accuracy
    metrics["accuracy"] = float(np.mean(y_true == y_pred))
    
    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # F1 score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    else:
        metrics["f1"] = 0.0
    
    # Probabilistic metrics (if probabilities provided)
    if y_prob is not None:
        try:
            # ROC AUC
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
            
            # Average Precision (PR AUC)
            metrics["ap"] = float(average_precision_score(y_true, y_prob))
            
            # Brier Score
            metrics["brier"] = float(brier_score_loss(y_true, y_prob))
            
            # Expected Calibration Error
            metrics["ece"] = float(expected_calibration_error(y_true, y_prob))
            
        except ValueError as e:
            # Handle case where all labels are the same class
            print(f"Warning: Could not compute probabilistic metrics: {e}")
            metrics["auroc"] = 0.5
            metrics["ap"] = float(y_true.mean())
            metrics["brier"] = float(np.mean((y_prob - y_true) ** 2))
            metrics["ece"] = 0.0
    
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                   output_path: Path, title: str = "ROC Curve") -> bool:
    """Plot ROC curve and save to file.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
        
    Returns:
        True if successful, False otherwise
    """
    try:
        plt.figure(figsize=(8, 6), dpi=100)
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Failed to plot ROC curve: {e}")
        return False


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                  output_path: Path, title: str = "Precision-Recall Curve") -> bool:
    """Plot Precision-Recall curve and save to file.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
        
    Returns:
        True if successful, False otherwise
    """
    try:
        plt.figure(figsize=(8, 6), dpi=100)
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        baseline = y_true.mean()
        
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Failed to plot PR curve: {e}")
        return False


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                          output_path: Path, title: str = "Calibration Curve") -> bool:
    """Plot calibration curve and save to file.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
        
    Returns:
        True if successful, False otherwise
    """
    try:
        plt.figure(figsize=(8, 6), dpi=100)
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10, strategy='uniform'
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2,
                label="Model")
        plt.plot([0, 1], [0, 1], "k:", linewidth=1, label="Perfectly calibrated")
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Failed to plot calibration curve: {e}")
        return False


def train_baseline_model(X: np.ndarray, y: np.ndarray, 
                        model_type: str = "logistic",
                        random_state: int = 42) -> Tuple[object, Dict[str, float]]:
    """Train a baseline model and return model + metrics.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model ("logistic" or "rf")
        random_state: Random seed
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == "logistic":
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # Compute metrics
    train_metrics = compute_metrics(y_train, y_pred_train, y_prob_train)
    test_metrics = compute_metrics(y_test, y_pred_test, y_prob_test)
    
    # Combine metrics
    metrics = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": X.shape[1],
        "model_type": model_type,
        "train_accuracy": train_metrics["accuracy"],
        "train_auroc": train_metrics["auroc"],
        "train_ap": train_metrics["ap"],
        "train_brier": train_metrics["brier"],
        "train_ece": train_metrics["ece"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auroc": test_metrics["auroc"], 
        "test_ap": test_metrics["ap"],
        "test_brier": test_metrics["brier"],
        "test_ece": test_metrics["ece"],
        # Main metrics (test performance)
        "accuracy": test_metrics["accuracy"],
        "auroc": test_metrics["auroc"],
        "ap": test_metrics["ap"],
        "brier": test_metrics["brier"],
        "ece": test_metrics["ece"],
        "timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }
    
    # Store test predictions for plotting
    model._test_data = {
        "y_true": y_test,
        "y_pred": y_pred_test,
        "y_prob": y_prob_test,
        "scaler": scaler
    }
    
    return model, metrics


def evaluate_model_and_save(data: pd.DataFrame, 
                           target_col: str = "target",
                           feature_cols: Optional[List[str]] = None,
                           model_type: str = "logistic",
                           output_dir: Path = Path("reports"),
                           random_state: int = 42) -> Dict[str, float]:
    """Train model, evaluate, and save metrics and figures.
    
    Args:
        data: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns (if None, auto-detect)
        model_type: Type of model to train
        output_dir: Directory to save outputs
        random_state: Random seed
        
    Returns:
        Dictionary of metrics
    """
    print(f"Training {model_type} model and generating evaluation reports...")
    
    # Prepare data
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [target_col, "plate_id", "well_row", "well_col", "compound_id"]
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found")
    
    # Clean data
    clean_data = data[[target_col] + feature_cols].dropna()
    if len(clean_data) < 10:
        raise ValueError("Insufficient clean data for training")
    
    X = clean_data[feature_cols].values
    y = clean_data[target_col].values
    
    print(f"  Training data: {len(clean_data)} samples, {len(feature_cols)} features")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Train model
    model, metrics = train_baseline_model(X, y, model_type, random_state)
    
    # Create output directories
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots using test data
    test_data = model._test_data
    y_true = test_data["y_true"]
    y_prob = test_data["y_prob"]
    
    print("  Generating evaluation plots...")
    
    # ROC curve
    plot_roc_curve(y_true, y_prob, 
                  figures_dir / "roc.png", 
                  f"ROC Curve - {model_type.upper()} Model")
    
    # PR curve  
    plot_pr_curve(y_true, y_prob,
                 figures_dir / "pr.png",
                 f"Precision-Recall Curve - {model_type.upper()} Model")
    
    # Calibration curve
    plot_calibration_curve(y_true, y_prob,
                          figures_dir / "calibration.png", 
                          f"Calibration Curve - {model_type.upper()} Model")
    
    # Save metrics
    metrics_path = output_dir / "model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  Saved metrics to {metrics_path}")
    print(f"  Test AUROC: {metrics['auroc']:.3f}")
    print(f"  Test AP: {metrics['ap']:.3f}")
    print(f"  Test Brier: {metrics['brier']:.3f}")
    print(f"  Test ECE: {metrics['ece']:.3f}")
    
    return metrics


if __name__ == "__main__":
    # CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eval.py <data_file.csv> [target_col] [model_type] [output_dir]")
        print("Model types: logistic, rf")
        sys.exit(1)
    
    data_file = sys.argv[1]
    target_col = sys.argv[2] if len(sys.argv) > 2 else "target"
    model_type = sys.argv[3] if len(sys.argv) > 3 else "logistic"
    output_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("reports")
    
    try:
        # Load data
        data = pd.read_csv(data_file)
        
        # Train and evaluate
        metrics = evaluate_model_and_save(
            data, target_col, model_type=model_type, 
            output_dir=output_dir, random_state=42
        )
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)