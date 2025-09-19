"""Run Cellpose segmentation on a directory of Cell Painting images."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from cellpose import models
from skimage import io
from tqdm import tqdm

from .utils import configure_logging, ensure_dir, find_image_triplets, safe_relative_path, set_seed

LOGGER = logging.getLogger("cp_tox.segment")


def load_stack(channels: Sequence[Path]) -> np.ndarray:
    stack = [io.imread(path).astype(np.float32) for path in channels]
    data = np.stack(stack, axis=-1)
    if data.max() > 0:
        data /= data.max()
    return data


def segment_image(
    image: np.ndarray,
    model: models.Cellpose,
    diameter: float | None,
    normalize: bool = False,
) -> np.ndarray:
    if normalize:
        image = (image - image.mean()) / (image.std() + 1e-6)
    masks, _, _, _ = model.eval(
        image,
        channels=[0, 0],
        diameter=diameter,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        do_3D=False,
    )
    return masks.astype(np.uint16)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with raw TIFFs")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store PNG masks")
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=["001", "002", "003"],
        help="Suffix tokens identifying channels per field (ordered DNA/actin/tubulin)",
    )
    parser.add_argument("--diameter", type=float, default=60.0, help="Expected cell diameter in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model_type", type=str, default="cyto2", help="Cellpose model type")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
    parser.add_argument("--normalize", action="store_true", help="Z-score normalize stacks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    set_seed(args.seed)

    triplets = find_image_triplets(args.input_dir, args.suffixes)
    if not triplets:
        LOGGER.error("No image triplets discovered in %s", safe_relative_path(args.input_dir))
        return 1

    ensure_dir(args.output_dir)
    model = models.Cellpose(model_type=args.model_type, gpu=False)
    LOGGER.info("Segmenting %d fields", len(triplets))

    for channels in tqdm(triplets, desc="Segment", unit="field"):
        stack = load_stack(channels)
        output_path = args.output_dir / (channels[0].stem + "_mask.png")
        if output_path.exists() and not args.overwrite:
            LOGGER.debug("Skipping existing mask %s", safe_relative_path(output_path))
            continue
        mask = segment_image(stack, model=model, diameter=args.diameter, normalize=args.normalize)
        io.imsave(output_path, mask, check_contrast=False)

    LOGGER.info("Segmentation complete")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
