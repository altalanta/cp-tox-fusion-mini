"""Compute basic QC metrics and generate a report for Cell Painting images."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import filters
from skimage.io import imread

from .extract_cp_features import CHANNEL_SUFFIXES, parse_metadata
from .utils import configure_logging, data_dir, ensure_dir, reports_dir, safe_relative_path

LOGGER = logging.getLogger("cp_tox.qc")


def compute_focus_metric(image: np.ndarray) -> float:
    grayscale = image.mean(axis=-1) if image.ndim == 3 else image
    lap = filters.laplace(grayscale.astype(np.float32))
    return float(lap.var())


def fit_illumination_gradient(image: np.ndarray) -> float:
    grayscale = image.mean(axis=-1) if image.ndim == 3 else image
    rows, cols = grayscale.shape
    yy, xx = np.mgrid[:rows, :cols]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(rows * cols)]
    coeffs, *_ = np.linalg.lstsq(A, grayscale.ravel(), rcond=None)
    gradient = np.hypot(coeffs[0], coeffs[1])
    return float(gradient)


def compute_debris_ratio(mask: np.ndarray, min_area: float = 50.0) -> float:
    labels, counts = np.unique(mask, return_counts=True)
    if labels.size == 0:
        return float("nan")
    valid = labels != 0
    if valid.sum() == 0:
        return float("nan")
    cell_sizes = counts[valid]
    debris = (cell_sizes < min_area).sum()
    return float(debris / max(len(cell_sizes), 1))


def load_channels(stem: str, image_dir: Path, suffixes: Sequence[str]) -> np.ndarray:
    arrays = []
    for suffix in suffixes:
        path = image_dir / f"{stem[:-3]}{suffix}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing channel {suffix} for {stem}")
        arrays.append(imread(path).astype(np.float32))
    stack = np.stack(arrays, axis=-1)
    if stack.max() > 0:
        stack /= stack.max()
    return stack


def qc_table(mask_dir: Path, image_dir: Path, suffixes: Sequence[str]) -> pd.DataFrame:
    records = []
    for mask_path in sorted(mask_dir.glob("*_mask.png")):
        stem = mask_path.stem.replace("_mask", "")
        meta = parse_metadata(stem)
        mask = imread(mask_path)
        image = load_channels(stem, image_dir, suffixes)

        labels = np.unique(mask)
        cell_count = int((labels != 0).sum())
        focus = compute_focus_metric(image)
        gradient = fit_illumination_gradient(image)
        debris_ratio = compute_debris_ratio(mask)

        records.append(
            {
                "plate": meta["plate"],
                "compound_id": meta["compound"],
                "well": meta["well"],
                "site": meta["site"],
                "z_plane": meta["z"],
                "cell_count": cell_count,
                "focus_variance": focus,
                "illumination_gradient": gradient,
                "debris_ratio": debris_ratio,
                "mask_path": safe_relative_path(mask_path),
            }
        )
    return pd.DataFrame.from_records(records)


def control_drift(df: pd.DataFrame, control_compounds: Sequence[str]) -> pd.DataFrame:
    qc_df = df.copy()
    qc_df["cell_count_flag"] = (qc_df["cell_count"] < 10) | (qc_df["cell_count"] > 3000)
    if control_compounds:
        controls = qc_df[qc_df["compound_id"].isin(control_compounds)]
        if not controls.empty:
            control_medians = controls.groupby("plate")["cell_count"].median().rename("control_median")
            qc_df = qc_df.merge(control_medians, on="plate", how="left")
            qc_df["control_median"].fillna(qc_df.groupby("plate")["cell_count"].transform("median"), inplace=True)
            qc_df["control_drift"] = qc_df["cell_count"] - qc_df["control_median"]
        else:
            qc_df["control_median"] = np.nan
            qc_df["control_drift"] = np.nan
    else:
        qc_df["control_median"] = np.nan
        qc_df["control_drift"] = np.nan
    return qc_df


def write_plots(df: pd.DataFrame, report_folder: Path) -> dict[str, str]:
    ensure_dir(report_folder)
    figures: dict[str, str] = {}

    plt.figure(figsize=(6, 4))
    df.groupby("well")["focus_variance"].median().sort_values().plot(kind="bar")
    plt.ylabel("Median focus variance")
    plt.tight_layout()
    focus_path = report_folder / "qc_focus_variance.png"
    plt.savefig(focus_path, dpi=200)
    plt.close()
    figures["focus"] = focus_path.name

    plt.figure(figsize=(6, 4))
    df.groupby("well")["cell_count"].mean().sort_values().plot(kind="bar")
    plt.ylabel("Mean cell count")
    plt.tight_layout()
    cell_path = report_folder / "qc_cell_counts.png"
    plt.savefig(cell_path, dpi=200)
    plt.close()
    figures["cell_counts"] = cell_path.name

    if "control_drift" in df.columns and df["control_drift"].notna().any():
        plt.figure(figsize=(6, 4))
        drift = df.groupby("plate")["control_drift"].median()
        drift.plot(marker="o")
        plt.ylabel("Control drift (cells)")
        plt.tight_layout()
        drift_path = report_folder / "qc_control_drift.png"
        plt.savefig(drift_path, dpi=200)
        plt.close()
        figures["control_drift"] = drift_path.name

    plt.figure(figsize=(6, 4))
    df.groupby("well")["illumination_gradient"].median().sort_values().plot(kind="bar")
    plt.ylabel("Median illumination gradient")
    plt.tight_layout()
    illum_path = report_folder / "qc_illumination_gradient.png"
    plt.savefig(illum_path, dpi=200)
    plt.close()
    figures["illumination"] = illum_path.name

    return figures


def write_report(df: pd.DataFrame, figures: dict[str, str], report_path: Path) -> None:
    lines = ["# Cell Painting QC Report", ""]
    summary = df[["focus_variance", "illumination_gradient", "debris_ratio", "cell_count"]].describe()
    lines.append("## Summary Statistics")
    lines.append(summary.to_markdown())
    lines.append("")

    lines.append("## QC Flags")
    flag_counts = df["cell_count_flag"].value_counts(dropna=False)
    lines.append(flag_counts.to_frame("count").to_markdown())
    lines.append("")

    for key, filename in figures.items():
        title = {
            "focus": "Focus Variance by Well",
            "cell_counts": "Cell Count by Well",
            "control_drift": "Control Drift per Plate",
            "illumination": "Illumination Gradient by Well",
        }[key]
        lines.append(f"## {title}")
        lines.append(f"![{title}]({filename})")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory with Cell Painting TIFFs")
    parser.add_argument("--mask_dir", type=Path, required=True, help="Directory with segmentation masks")
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir("processed") / "qc_metrics.parquet",
        help="Path to the QC metrics parquet output",
    )
    parser.add_argument(
        "--control_compounds",
        nargs="*",
        default=["000002"],
        help="Compound IDs treated as plate controls (optional)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    df = qc_table(args.mask_dir, args.image_dir, CHANNEL_SUFFIXES)
    df = control_drift(df, args.control_compounds)

    ensure_dir(args.output.parent)
    df.to_parquet(args.output, index=False)
    LOGGER.info("Wrote QC metrics to %s", safe_relative_path(args.output))

    report_folder = reports_dir()
    figures = write_plots(df, report_folder)
    report_path = report_folder / "qc_report.md"
    write_report(df, figures, report_path)
    LOGGER.info("Wrote QC report to %s", safe_relative_path(report_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
