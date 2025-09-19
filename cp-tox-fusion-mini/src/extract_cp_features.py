"""Feature extraction utilities for segmented Cell Painting images."""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pycytominer.cyto_utils.aggregate import Aggregate
from skimage import feature, measure
from skimage.io import imread

from .utils import configure_logging, data_dir, ensure_dir, safe_relative_path

LOGGER = logging.getLogger("cp_tox.extract_features")

METADATA_PATTERN = re.compile(
    r"(?P<plate>Week\d+_\d+)_AS_(?P<compound>\d+)_(?P<channel>\d{3})_"
    r"(?P<well>[A-H]\d{2})f(?P<site>\d{2})d(?P<z>\d)"
)


@dataclass
class ImageBundle:
    """Container for a multi-channel image and associated metadata."""

    plate: str
    compound: str
    well: str
    site: str
    z: str
    stack: np.ndarray
    mask: np.ndarray


CHANNEL_SUFFIXES = ("001", "002", "003")


def parse_metadata(stem: str) -> Dict[str, str]:
    match = METADATA_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Could not parse metadata from {stem}")
    return match.groupdict()


def load_bundle(mask_path: Path, image_dir: Path) -> ImageBundle:
    stem = mask_path.stem.replace("_mask", "")
    meta = parse_metadata(stem)
    channels = []
    for suffix in CHANNEL_SUFFIXES:
        image_path = image_dir / f"{stem[:-3]}{suffix}.tif"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing channel {suffix} for {stem}")
        channels.append(imread(image_path).astype(np.float32))
    stack = np.stack(channels, axis=-1)
    if stack.max() > 0:
        stack /= stack.max()
    mask = imread(mask_path).astype(np.int32)
    return ImageBundle(
        plate=meta["plate"],
        compound=meta["compound"],
        well=meta["well"],
        site=meta["site"],
        z=meta["z"],
        stack=stack,
        mask=mask,
    )


def compute_texture(patch: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if patch.ndim != 2:
        raise ValueError("Texture expects 2D patch")
    if mask.sum() < 20:
        return {"contrast": float("nan"), "homogeneity": float("nan"), "dissimilarity": float("nan")}

    masked = np.where(mask, patch, 0)
    scaled = np.clip(masked * 255, 0, 255).astype(np.uint8)
    glcm = feature.graycomatrix(scaled, distances=[1], angles=[0], symmetric=True, normed=True)
    return {
        "contrast": float(feature.graycoprops(glcm, "contrast")[0, 0]),
        "homogeneity": float(feature.graycoprops(glcm, "homogeneity")[0, 0]),
        "dissimilarity": float(feature.graycoprops(glcm, "dissimilarity")[0, 0]),
    }


def per_cell_features(bundle: ImageBundle) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    channels = {
        "dna": bundle.stack[..., 0],
        "actin": bundle.stack[..., 1],
        "tubulin": bundle.stack[..., 2],
    }

    for region in measure.regionprops(bundle.mask, intensity_image=bundle.stack[..., 0]):
        record: Dict[str, float] = {
            "cell_id": int(region.label),
            "area": float(region.area),
            "perimeter": float(region.perimeter),
            "eccentricity": float(region.eccentricity),
            "solidity": float(region.solidity),
            "dna_mean_intensity": float(region.mean_intensity),
        }
        for name, channel in channels.items():
            patch = channel[region.slice]
            mask = region.image
            pixels = patch[mask]
            record[f"{name}_intensity_mean"] = float(pixels.mean()) if pixels.size else float("nan")
            record[f"{name}_intensity_std"] = float(pixels.std()) if pixels.size else float("nan")
            record[f"{name}_intensity_max"] = float(pixels.max()) if pixels.size else float("nan")
            record[f"{name}_intensity_min"] = float(pixels.min()) if pixels.size else float("nan")
            texture = compute_texture(patch, mask)
            for metric, value in texture.items():
                record[f"{name}_texture_{metric}"] = value
        record["plate"] = bundle.plate
        record["compound_id"] = bundle.compound
        record["well"] = bundle.well
        record["site"] = bundle.site
        record["z_plane"] = bundle.z
        records.append(record)

    return pd.DataFrame.from_records(records)


def aggregate_profiles(cells: pd.DataFrame) -> pd.DataFrame:
    aggregator = Aggregate(population_df=cells, strata=["plate", "compound_id", "well"], operation="median")
    profile_df = aggregator.aggregate()
    profile_df.reset_index(inplace=True)
    counts = cells.groupby(["plate", "compound_id", "well"]).size().rename("cell_count").reset_index()
    profile_df = profile_df.merge(counts, on=["plate", "compound_id", "well"], how="left")
    return profile_df


def extract_features(mask_dir: Path, image_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_cell_records: List[pd.DataFrame] = []
    for mask_path in sorted(mask_dir.glob("*_mask.png")):
        bundle = load_bundle(mask_path, image_dir)
        cell_df = per_cell_features(bundle)
        cell_df["image_path"] = safe_relative_path(mask_path)
        per_cell_records.append(cell_df)

    if not per_cell_records:
        raise RuntimeError("No mask files were found; run segmentation first")

    cells = pd.concat(per_cell_records, ignore_index=True)
    profiles = aggregate_profiles(cells)
    return cells, profiles


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory with raw TIFF images")
    parser.add_argument("--mask_dir", type=Path, required=True, help="Directory with Cellpose masks")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=data_dir("processed"),
        help="Directory where feature tables will be saved",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    ensure_dir(args.output_dir)
    try:
        cells, profiles = extract_features(args.mask_dir, args.image_dir)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Feature extraction failed: %s", exc)
        return 1

    per_cell_path = args.output_dir / "cp_single_cell.parquet"
    profile_path = args.output_dir / "cp_features.parquet"
    cells.to_parquet(per_cell_path, index=False)
    profiles.to_parquet(profile_path, index=False)
    LOGGER.info("Wrote %s and %s", safe_relative_path(per_cell_path), safe_relative_path(profile_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
