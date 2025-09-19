"""Download a lightweight Tox21 assay table with SMILES and labels."""
from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .utils import configure_logging, data_dir, ensure_dir, safe_relative_path

LOGGER = logging.getLogger("cp_tox.download_tox21")

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
RAW_FILENAME = "tox21.csv.gz"
PROCESSED_FILENAME = "tox21_mini.csv"
MAX_ROWS = 5000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir("raw", "tox21"),
        help="Output directory for the downloaded files",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=MAX_ROWS,
        help="Maximum number of rows to retain in the CSV subset",
    )
    return parser.parse_args(argv)


def download(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    LOGGER.info("Downloading %s", url)
    with urlopen(url) as response, destination.open("wb") as fh:
        fh.write(response.read())


def load_subset(path: Path, max_rows: int) -> pd.DataFrame:
    LOGGER.info("Loading Tox21 table from %s", safe_relative_path(path))
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh)
    keep_cols = ["mol_id", "smiles", "toxic", "nr-ahr", "nr-aromatase", "sr-p53"]
    available = [col for col in keep_cols if col in df.columns]
    subset = df.loc[:, available].dropna(subset=["smiles"]).head(max_rows)
    subset = subset.rename(columns={"smiles": "SMILES", "mol_id": "compound_id"})
    LOGGER.info("Subset shape: %s", subset.shape)
    return subset


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    raw_dir = args.output
    ensure_dir(raw_dir)
    raw_path = raw_dir / RAW_FILENAME
    processed_path = raw_dir / PROCESSED_FILENAME

    if not raw_path.exists():
        try:
            download(TOX21_URL, raw_path)
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Unable to download Tox21 dataset: %s", exc)
            return 1
    else:
        LOGGER.info("Using cached %s", safe_relative_path(raw_path))

    try:
        subset = load_subset(raw_path, args.rows)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed to parse Tox21 CSV: %s", exc)
        return 1

    subset.to_csv(processed_path, index=False)
    LOGGER.info("Saved subset to %s", safe_relative_path(processed_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
