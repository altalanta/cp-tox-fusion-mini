"""Align Cell Painting and chemical descriptors with consistent splits."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .utils import configure_logging, data_dir, ensure_dir, save_json, safe_relative_path

LOGGER = logging.getLogger("cp_tox.fuse")

CHEM_META_COLUMNS = {"SMILES", "compound_id"}


def viability_labels(cp_df: pd.DataFrame, control_ids: List[str]) -> pd.DataFrame:
    df = cp_df.copy()
    controls = df[df["compound_id"].isin(control_ids)]
    control_median = controls.groupby("plate")["cell_count"].median()
    df["control_median"] = df["plate"].map(control_median)
    df["control_median"].fillna(df.groupby("plate")["cell_count"].transform("median"), inplace=True)
    df["viability_ratio"] = df["cell_count"] / df["control_median"].replace(0, float("nan"))
    df["viability_ratio"].replace([float("inf"), -float("inf")], float("nan"), inplace=True)
    df["viability_label"] = (df["viability_ratio"] < 0.8).astype(int)
    return df


def prefix_features(df: pd.DataFrame, prefix: str, exclude: set[str]) -> pd.DataFrame:
    features = df.drop(columns=[col for col in df.columns if col in exclude])
    features = features.add_prefix(prefix)
    return pd.concat([df[list(exclude & set(df.columns))], features], axis=1)


def load_mapping(mapping_path: Path | None, cp_ids: List[str], chem_ids: List[str]) -> pd.DataFrame:
    if mapping_path and mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        expected_cols = {"cp_compound_id", "chem_compound_id"}
        if not expected_cols.issubset(mapping.columns):
            raise ValueError(f"Mapping file must contain columns {expected_cols}")
        LOGGER.info("Loaded mapping from %s", safe_relative_path(mapping_path))
        return mapping

    LOGGER.warning("No mapping provided – generating heuristic mapping for demonstration")
    chem_unique = pd.unique(chem_ids)
    if len(chem_unique) < len(cp_ids):
        raise ValueError("Not enough chemical compounds to map onto Cell Painting IDs")
    pairs = list(zip(cp_ids, chem_unique[: len(cp_ids)]))
    mapping = pd.DataFrame(pairs, columns=["cp_compound_id", "chem_compound_id"])
    return mapping


def group_split(df: pd.DataFrame, group_col: str, random_state: int = 42) -> Dict[str, List[str]]:
    if df.empty:
        return {"train": [], "val": [], "test": []}
    groups = df[group_col]
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=random_state)
    train_idx, rest_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx]
    rest_df = df.iloc[rest_idx]
    if rest_df.empty or rest_df[group_col].nunique() < 2:
        val_df = rest_df
        test_df = pd.DataFrame(columns=df.columns)
    else:
        splitter_val = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state)
        val_idx, test_idx = next(splitter_val.split(rest_df, groups=rest_df[group_col]))
        val_df = rest_df.iloc[val_idx]
        test_df = rest_df.iloc[test_idx]
    return {
        "train": sorted(train_df[group_col].unique().tolist()),
        "val": sorted(val_df[group_col].unique().tolist()),
        "test": sorted(test_df[group_col].unique().tolist()),
    }


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cp_features",
        type=Path,
        default=data_dir("processed", "cp_features.parquet"),
        help="Path to aggregated Cell Painting features",
    )
    parser.add_argument(
        "--chem_features",
        type=Path,
        default=data_dir("processed", "rdkit_features.parquet"),
        help="Path to RDKit descriptor table",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=data_dir("processed", "compound_mapping.csv"),
        help="Optional compound mapping CSV (cp_compound_id, chem_compound_id)",
    )
    parser.add_argument(
        "--control_ids",
        nargs="*",
        default=["000002"],
        help="Compound IDs treated as plate controls for viability normalization",
    )
    parser.add_argument(
        "--tox_target",
        type=str,
        default="sr-p53",
        help="Tox21 target column to use as binary label",
    )
    parser.add_argument("--output_dir", type=Path, default=data_dir("processed"))
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    cp_df = pd.read_parquet(args.cp_features)
    chem_df = pd.read_parquet(args.chem_features)

    cp_df = viability_labels(cp_df, args.control_ids)
    cp_prefixed = prefix_features(
        cp_df,
        prefix="cp_",
        exclude={"plate", "compound_id", "well", "viability_label", "viability_ratio", "cell_count"},
    )

    chem_meta_cols = [col for col in CHEM_META_COLUMNS if col in chem_df.columns]
    if args.tox_target not in chem_df.columns:
        candidates = [
            col
            for col in chem_df.columns
            if col not in CHEM_META_COLUMNS and (col.startswith("nr-") or col.startswith("sr-"))
        ]
        if candidates:
            LOGGER.warning("Target %s missing; falling back to %s", args.tox_target, candidates[0])
            args.tox_target = candidates[0]
        else:
            raise ValueError("No toxicity label columns found in RDKit features")
    chem_prefixed = prefix_features(chem_df, prefix="chem_", exclude=set(chem_meta_cols + [args.tox_target]))
    if args.tox_target in chem_prefixed.columns:
        chem_prefixed.rename(columns={args.tox_target: "toxicity_label"}, inplace=True)

    mapping = load_mapping(
        args.mapping if args.mapping.exists() else None,
        sorted(cp_df["compound_id"].unique()),
        chem_df["compound_id"].dropna().tolist(),
    )
    ensure_dir(args.output_dir)
    mapping.to_csv(args.output_dir / "compound_mapping.csv", index=False)

    cp_dataset = cp_prefixed.copy()
    chem_dataset = chem_prefixed.copy()

    fusion = (
        cp_dataset.merge(mapping, left_on="compound_id", right_on="cp_compound_id", how="inner")
        .merge(
            chem_dataset,
            left_on="chem_compound_id",
            right_on="compound_id",
            suffixes=("", "_chem"),
            how="inner",
        )
    )

    if fusion.empty:
        LOGGER.warning("Fusion dataset is empty – provide an explicit mapping for overlap")

    cp_dataset.to_parquet(args.output_dir / "cp_dataset.parquet", index=False)
    chem_dataset.to_parquet(args.output_dir / "chem_dataset.parquet", index=False)
    if not fusion.empty:
        fusion.to_parquet(args.output_dir / "fusion_dataset.parquet", index=False)

    splits = {
        "cp": group_split(cp_dataset, "compound_id"),
        "chem": group_split(chem_dataset, "compound_id"),
        "fusion": group_split(fusion, "compound_id") if not fusion.empty else {"train": [], "val": [], "test": []},
    }
    save_json(splits, args.output_dir / "splits.json")
    LOGGER.info("Saved datasets and splits to %s", safe_relative_path(args.output_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
