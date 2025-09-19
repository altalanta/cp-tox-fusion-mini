"""Utility helpers for the cp-tox-fusion-mini project."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger("cp_tox")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the project."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root() -> Path:
    """Return the absolute path to the project root directory."""

    return Path(__file__).resolve().parents[1]


def data_dir(*parts: str) -> Path:
    """Return a path rooted inside the ``data`` directory."""

    return project_root() / "data" / Path(*parts)


def reports_dir(*parts: str) -> Path:
    """Return a path rooted inside the ``reports`` directory."""

    return project_root() / "reports" / Path(*parts)


def ensure_dir(path: Path) -> None:
    """Create ``path`` if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (if installed) RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover - torch optional
        LOGGER.debug("Torch not available for seeding", exc_info=True)


def find_image_triplets(image_dir: Path, suffixes: Sequence[str]) -> List[List[Path]]:
    """Identify sets of images representing the same field across channels."""

    grouped: dict[str, List[Path]] = {}
    for file in sorted(image_dir.glob("*.tif")):
        stem = file.stem
        for suffix in suffixes:
            if stem.endswith(suffix):
                key = stem[: -len(suffix)]
                grouped.setdefault(key, [None] * len(suffixes))
                grouped[key][suffixes.index(suffix)] = file
                break

    triplets = [paths for paths in grouped.values() if None not in paths]
    if not triplets:
        LOGGER.warning("No multi-channel image triplets discovered in %s", image_dir)
    return triplets


def load_json(path: Path) -> dict:
    """Load JSON from ``path`` and return the parsed object."""

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: dict, path: Path, indent: int = 2) -> None:
    """Serialize ``obj`` to JSON at ``path``."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, sort_keys=True)


def chunked(iterable: Sequence[str] | Iterable[str], size: int) -> Iterable[List[str]]:
    """Yield chunks from ``iterable`` of length ``size`` (last chunk shorter)."""

    chunk: List[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def safe_relative_path(path: Path, base: Optional[Path] = None) -> str:
    """Return a relative path string without raising when outside ``base``."""

    base = base or project_root()
    try:
        return str(path.resolve().relative_to(base))
    except ValueError:
        return str(path.resolve())


__all__ = [
    "configure_logging",
    "project_root",
    "data_dir",
    "reports_dir",
    "ensure_dir",
    "set_seed",
    "find_image_triplets",
    "load_json",
    "save_json",
    "chunked",
    "safe_relative_path",
]
