"""Generate RDKit descriptors for the Tox21 subset."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from .utils import configure_logging, data_dir, ensure_dir, safe_relative_path

LOGGER = logging.getLogger("cp_tox.descriptors")

PHYS_CHEM_FEATURES = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "TPSA": Descriptors.TPSA,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
    "SlogP_VSA1": Descriptors.SlogP_VSA1,
    "SMR_VSA1": Descriptors.SMR_VSA1,
    "qed": Descriptors.qed,
}


def morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_features(smiles: str) -> dict[str, float | int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    mol = Chem.AddHs(mol)

    features: dict[str, float | int] = {}
    for name, func in PHYS_CHEM_FEATURES.items():
        if func is None:
            continue
        try:
            features[name] = float(func(mol))
        except Exception:  # pragma: no cover
            features[name] = float("nan")

    fp = morgan_fingerprint(mol)
    for idx, value in enumerate(fp):
        features[f"morgan_{idx:04d}"] = int(value)
    return features


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir("raw", "tox21", "tox21_mini.csv"),
        help="Path to the curated Tox21 CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir("processed", "rdkit_features.parquet"),
        help="Output path for descriptor table",
    )
    parser.add_argument("--max_rows", type=int, default=5000, help="Optional limit on rows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    df = pd.read_csv(args.input)
    if args.max_rows:
        df = df.head(args.max_rows)

    records = []
    failed = 0
    for _, row in df.iterrows():
        smiles = row["SMILES"]
        try:
            feats = smiles_to_features(smiles)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Skipping molecule with invalid SMILES: %s (%s)", smiles, exc)
            failed += 1
            continue
        feats["SMILES"] = smiles
        for col, value in row.items():
            if col != "SMILES":
                feats[col] = value
        records.append(feats)

    if not records:
        LOGGER.error("No descriptors computed; aborting")
        return 1

    result = pd.DataFrame.from_records(records)
    ensure_dir(args.output.parent)
    result.to_parquet(args.output, index=False)
    LOGGER.info(
        "Saved %d molecules with descriptors to %s (skipped %d)",
        len(result),
        safe_relative_path(args.output),
        failed,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
