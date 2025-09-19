"""Generate error analysis artifacts for cp-tox-fusion-mini models."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from .utils import configure_logging, data_dir, ensure_dir, reports_dir, safe_relative_path

LOGGER = logging.getLogger("cp_tox.error_analysis")

PLOT_CONFIG = {
    "image_confusion": "image_confusion.png",
    "chem_confusion": "chem_confusion.png",
    "fusion_confusion": "fusion_confusion.png",
    "image_calibration": "image_calibration.png",
    "chem_calibration": "chem_calibration.png",
    "fusion_calibration": "fusion_calibration.png",
    "residuals_qc": "residuals_vs_qc.png",
    "fusion_gain": "fusion_gain.png",
}


def load_predictions(predictions_dir: Path, prefix: str) -> pd.DataFrame:
    if prefix == "image":
        path = predictions_dir / "image_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing image predictions at {path}")
        return pd.read_parquet(path)

    frames = []
    for file in predictions_dir.glob(f"{prefix}_predictions_*.parquet"):
        frames.append(pd.read_parquet(file))
    if not frames:
        LOGGER.warning("No predictions found for prefix %s", prefix)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, title: str, output_path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=min(5, len(np.unique(y_prob))))
    plt.figure(figsize=(4, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def stratify_by_bin(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    grouped = df.groupby(column)
    records = []
    for level, group in grouped:
        if group.empty:
            continue
        preds = (group["probability"] >= 0.5).astype(int)
        accuracy = (preds == group[target]).mean()
        records.append({column: level, "count": int(len(group)), "accuracy": float(accuracy)})
    return pd.DataFrame(records)


def assign_dose_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = []
    for value in df["viability_ratio"].fillna(1.0):
        if value >= 0.9:
            bins.append("low")
        elif value >= 0.6:
            bins.append("medium")
        else:
            bins.append("high")
    df = df.copy()
    df["dose_bin"] = bins
    return df


def fusion_gain_table(image_df: pd.DataFrame, chem_df: pd.DataFrame, fusion_df: pd.DataFrame) -> pd.DataFrame:
    if fusion_df.empty:
        return pd.DataFrame()
    agg = {}
    for name, frame in {"image": image_df, "chem": chem_df, "fusion": fusion_df}.items():
        if frame.empty:
            continue
        aggregated = frame.groupby("compound_id").agg({"probability": "mean", "true_label": "mean"})
        aggregated["prediction"] = (aggregated["probability"] >= 0.5).astype(int)
        agg[name] = aggregated
    if "fusion" not in agg:
        return pd.DataFrame()
    combined = agg["fusion"]["prediction"].to_frame("fusion_pred")
    for name, table in agg.items():
        if name == "fusion":
            continue
        combined = combined.join(table[["prediction"]].rename(columns={"prediction": f"{name}_pred"}), how="left")
    combined = combined.join(agg["fusion"]["true_label"].rename("true_label"))
    combined.dropna(subset=["true_label"], inplace=True)
    combined["fusion_correct"] = (combined["fusion_pred"] == combined["true_label"]).astype(int)
    for name in ["image", "chem"]:
        if f"{name}_pred" in combined:
            combined[f"{name}_correct"] = (combined[f"{name}_pred"] == combined["true_label"]).astype(int)
    combined["fusion_beats_image"] = combined.get("fusion_correct", 0) - combined.get("image_correct", 0)
    combined["fusion_beats_chem"] = combined.get("fusion_correct", 0) - combined.get("chem_correct", 0)
    summary = combined[["fusion_beats_image", "fusion_beats_chem"]]
    summary.loc["total", :] = summary.sum()
    return summary


def build_report(
    metrics: Dict,
    tables: Dict[str, pd.DataFrame],
    plots: Dict[str, str],
    report_path: Path,
    notes: List[str],
) -> None:
    lines: List[str] = ["# CP-Tox Fusion Mini Error Analysis", ""]
    lines.append("## Model Metrics (Test)")
    for model_name, model_metrics in metrics.items():
        if not model_metrics:
            lines.append(f"- {model_name}: no evaluation available")
            continue
        test_metrics = model_metrics.get("test", {})
        metrics_str = ", ".join(
            f"{key}={value:.3f}" for key, value in test_metrics.items() if not np.isnan(value)
        )
        lines.append(f"- **{model_name.title()}**: {metrics_str}")
    lines.append("")

    for label, table in tables.items():
        if table.empty:
            continue
        lines.append(f"## {label}")
        lines.append(table.to_markdown(index=True))
        lines.append("")

    lines.append("## Plots")
    for description, filename in plots.items():
        if not filename:
            continue
        title = description.replace("_", " ").title()
        lines.append(f"### {title}")
        lines.append(f"![{title}]({filename})")
        lines.append("")

    if notes:
        lines.append("## Notes")
        for note in notes:
            lines.append(f"- {note}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, default=data_dir("processed"))
    parser.add_argument("--predictions_dir", type=Path, default=data_dir("processed", "predictions"))
    parser.add_argument("--reports_dir", type=Path, default=reports_dir())
    parser.add_argument("--qc_path", type=Path, default=data_dir("processed", "qc_metrics.parquet"))
    parser.add_argument("--metrics_path", type=Path, default=reports_dir() / "model_metrics.json")
    parser.add_argument("--control_ids", nargs="*", default=["000002"])
    return parser.parse_args()


def main(argv: List[str] | None = None) -> int:
    args = parse_args()
    configure_logging()
    ensure_dir(args.reports_dir)

    metrics: Dict = {}
    if args.metrics_path.exists():
        metrics = json.loads(args.metrics_path.read_text())

    image_preds = load_predictions(args.predictions_dir, "image")
    chem_preds = load_predictions(args.predictions_dir, "chem")
    fusion_preds = load_predictions(args.predictions_dir, "fusion")

    cp_dataset = pd.read_parquet(args.processed_dir / "cp_dataset.parquet")
    cp_dataset = assign_dose_bins(cp_dataset)
    cp_dataset["moa"] = np.where(
        cp_dataset["compound_id"].isin(args.control_ids), "control", "perturbation"
    )

    qc_df = pd.read_parquet(args.qc_path) if args.qc_path.exists() else pd.DataFrame()

    image_test = image_preds[image_preds["split"] == "test"] if not image_preds.empty else pd.DataFrame()
    if not image_test.empty:
        image_join = image_test.merge(cp_dataset, on=["compound_id", "well"], how="left")
        if not qc_df.empty:
            image_join = image_join.merge(
                qc_df,
                on=["compound_id", "well", "site"],
                how="left",
                suffixes=("", "_qc"),
            )
        y_true = image_join["true_label"].to_numpy()
        y_pred = (image_join["probability"] >= 0.5).astype(int).to_numpy()
        plot_confusion(y_true, y_pred, "Image Model", args.reports_dir / PLOT_CONFIG["image_confusion"])
        plot_calibration(
            y_true,
            image_join["probability"].to_numpy(),
            "Image Model",
            args.reports_dir / PLOT_CONFIG["image_calibration"],
        )
        dose_table = stratify_by_bin(image_join, "dose_bin", "true_label")
        moa_table = stratify_by_bin(image_join, "moa", "true_label")
        residuals = image_join[["focus_variance", "illumination_gradient", "probability", "true_label"]].dropna()
        if not residuals.empty:
            residuals["residual"] = residuals["probability"] - residuals["true_label"]
            plt.figure(figsize=(5, 4))
            plt.scatter(residuals["focus_variance"], residuals["residual"], alpha=0.6, label="Focus variance")
            plt.scatter(residuals["illumination_gradient"], residuals["residual"], alpha=0.6, label="Illumination gradient")
            plt.legend()
            plt.xlabel("QC metric value")
            plt.ylabel("Residual (prob - truth)")
            plt.tight_layout()
            plt.savefig(args.reports_dir / PLOT_CONFIG["residuals_qc"], dpi=200)
            plt.close()
    else:
        dose_table = pd.DataFrame()
        moa_table = pd.DataFrame()

    chem_test = chem_preds[chem_preds["split"] == "test"] if not chem_preds.empty else pd.DataFrame()
    if not chem_test.empty:
        y_true_chem = chem_test["true_label"].to_numpy()
        y_pred_chem = (chem_test["probability"] >= 0.5).astype(int).to_numpy()
        plot_confusion(
            y_true_chem,
            y_pred_chem,
            "Chem Model",
            args.reports_dir / PLOT_CONFIG["chem_confusion"],
        )
        plot_calibration(
            y_true_chem,
            chem_test["probability"].to_numpy(),
            "Chem Model",
            args.reports_dir / PLOT_CONFIG["chem_calibration"],
        )

    fusion_test = fusion_preds[fusion_preds["split"] == "test"] if not fusion_preds.empty else pd.DataFrame()
    if not fusion_test.empty:
        y_true_fusion = fusion_test["true_label"].to_numpy()
        y_pred_fusion = (fusion_test["probability"] >= 0.5).astype(int).to_numpy()
        plot_confusion(
            y_true_fusion,
            y_pred_fusion,
            "Fusion Model",
            args.reports_dir / PLOT_CONFIG["fusion_confusion"],
        )
        plot_calibration(
            y_true_fusion,
            fusion_test["probability"].to_numpy(),
            "Fusion Model",
            args.reports_dir / PLOT_CONFIG["fusion_calibration"],
        )

    fusion_table = fusion_gain_table(image_test, chem_test, fusion_test)
    if not fusion_table.empty:
        fusion_table.to_markdown(args.reports_dir / "fusion_gain_table.md")
        fusion_table.sum().plot(kind="bar")
        plt.ylabel("Count of compounds")
        plt.tight_layout()
        plt.savefig(args.reports_dir / PLOT_CONFIG["fusion_gain"], dpi=200)
        plt.close()

    tables = {
        "Dose-Stratified Accuracy": dose_table,
        "MoA-Stratified Accuracy": moa_table,
        "Fusion Helps": fusion_table,
    }
    plots = {
        "image_confusion": PLOT_CONFIG["image_confusion"] if not image_test.empty else "",
        "chem_confusion": PLOT_CONFIG["chem_confusion"] if not chem_test.empty else "",
        "fusion_confusion": PLOT_CONFIG["fusion_confusion"] if not fusion_test.empty else "",
        "image_calibration": PLOT_CONFIG["image_calibration"] if not image_test.empty else "",
        "chem_calibration": PLOT_CONFIG["chem_calibration"] if not chem_test.empty else "",
        "fusion_calibration": PLOT_CONFIG["fusion_calibration"] if not fusion_test.empty else "",
        "residuals_qc": PLOT_CONFIG["residuals_qc"] if not image_test.empty else "",
        "fusion_gain": PLOT_CONFIG["fusion_gain"] if not fusion_table.empty else "",
    }

    notes = []
    if fusion_table.empty:
        notes.append("Fusion dataset did not produce overlapping compounds; add mapping for richer analysis.")
    if qc_df.empty:
        notes.append("QC metrics not available; residual vs QC plot omitted.")

    report_path = args.reports_dir / "cp-tox-mini_report.md"
    build_report(metrics, tables, plots, report_path, notes)
    LOGGER.info("Wrote error analysis report to %s", safe_relative_path(report_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
