"""Download a curated BBBC021 Cell Painting subset for the mini workflow."""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.request import urlopen

from tqdm import tqdm

from .utils import configure_logging, data_dir, ensure_dir, safe_relative_path

LOGGER = logging.getLogger("cp_tox.download_bbbc021")


@dataclass(frozen=True)
class RemoteFile:
    """Descriptor for a remote file to download."""

    url: str
    sha256: str | None
    relpath: str

    def destination(self, root: Path) -> Path:
        return root / self.relpath


MANIFEST: tuple[RemoteFile, ...] = (
    RemoteFile(
        url="https://data.broadinstitute.org/bbbc/BBBC021/MCF7/Week1_22123/Week1_22123_AS_000002_001_A02f00d0.tif",
        sha256="16e5226e0b77653d9c707113683f1361f81306bca7999bee8e1cf9cedf7629d8",
        relpath="Week1_22123/Week1_22123_AS_000002_001_A02f00d0.tif",
    ),
    RemoteFile(
        url="https://data.broadinstitute.org/bbbc/BBBC021/MCF7/Week1_22123/Week1_22123_AS_000123_001_C07f00d0.tif",
        sha256="7dee2b6726376fbf1a8c8f4702f855d81d8348a52a66a99d91ad10b30be362c6",
        relpath="Week1_22123/Week1_22123_AS_000123_001_C07f00d0.tif",
    ),
    RemoteFile(
        url="https://data.broadinstitute.org/bbbc/BBBC021/MCF7/Week2_22026/Week2_22026_AS_000345_001_F05f00d0.tif",
        sha256="c6e898661a64c98870b46cf75c3f7b60f7af512e27945fe75357c9e775c8735f",
        relpath="Week2_22026/Week2_22026_AS_000345_001_F05f00d0.tif",
    ),
    RemoteFile(
        url="https://data.broadinstitute.org/bbbc/BBBC021/MCF7/Week1_22123/Week1_22123_AS_000002_002_A02f00d0.tif",
        sha256="d6de8b4e9c629c8fb16183fbda45074bce5c7674313c3d1d754477c3df4333c7",
        relpath="Week1_22123/Week1_22123_AS_000002_002_A02f00d0.tif",
    ),
    RemoteFile(
        url="https://data.broadinstitute.org/bbbc/BBBC021/MCF7/Week1_22123/Week1_22123_AS_000123_002_C07f00d0.tif",
        sha256="76b12c7c58f7453239fcfba6d42ee7bb1fcbc401d56cfa68c802d082980d9073",
        relpath="Week1_22123/Week1_22123_AS_000123_002_C07f00d0.tif",
    ),
    RemoteFile(
        url="https://data.broadinstitute.org/bbbc/BBBC021/MCF7/Week2_22026/Week2_22026_AS_000345_002_F05f00d0.tif",
        sha256="b2bdf470b095e3cf9f3e2274fc48e6ad0793750114b3f81ed8460fd36c36b343",
        relpath="Week2_22026/Week2_22026_AS_000345_002_F05f00d0.tif",
    ),
    RemoteFile(
        url="https://data.broadinstitute.org/bbbc/BBBC021/MCF7/Week1_22123/Week1_22123_AS_000002_003_A02f00d0.tif",
        sha256="4d83b402cc9cf1db292179c998c01d2ea8adbbaffe4a132daf3c8ca218996dca",
        relpath="Week1_22123/Week1_22123_AS_000002_003_A02f00d0.tif",
    ),
    RemoteFile(
        url="https://data.broadinstitute.org/bbbc/BBBC021/MCF7/Week1_22123/Week1_22123_AS_000123_003_C07f00d0.tif",
        sha256="c4576dd6a171b5cf1069ec356168dde4efd499c0f79d63a552ff65e3ba903d11",
        relpath="Week1_22123/Week1_22123_AS_000123_003_C07f00d0.tif",
    ),
    RemoteFile(
        url="https://data.broadinstitute.org/bbbc/BBBC021/MCF7/Week2_22026/Week2_22026_AS_000345_003_F05f00d0.tif",
        sha256="8c5a1fd66589a219c2db299d94f036f2008a53a4836bb4c1a562646b4c13f074",
        relpath="Week2_22026/Week2_22026_AS_000345_003_F05f00d0.tif",
    ),
)


ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2"}


def iter_members(archive: Path) -> Iterator[Path]:
    if archive.suffix == ".zip":
        LOGGER.info("Extracting ZIP archive %s", safe_relative_path(archive))
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(archive.parent)
        yield from archive.parent.glob("**/*.tif")
    elif archive.suffix in {".tar", ".gz", ".bz2", ".tgz", ".tar.gz", ".tar.bz2"}:
        LOGGER.info("Extracting TAR archive %s", safe_relative_path(archive))
        with tarfile.open(archive) as tf:
            tf.extractall(archive.parent)
        yield from archive.parent.glob("**/*.tif")
    else:
        yield archive


def sha256sum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(remote: RemoteFile, root: Path) -> Path:
    destination = remote.destination(root)
    ensure_dir(destination.parent)

    if destination.exists() and remote.sha256:
        existing_hash = sha256sum(destination)
        if existing_hash == remote.sha256:
            LOGGER.info("Skipping existing %s", safe_relative_path(destination))
            return destination
        LOGGER.warning(
            "Hash mismatch for %s (expected %s, found %s); re-downloading",
            safe_relative_path(destination),
            remote.sha256,
            existing_hash,
        )

    LOGGER.info("Downloading %s", remote.url)
    with urlopen(remote.url) as response, destination.open("wb") as fh:
        total = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as bar:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
                bar.update(len(chunk))

    if remote.sha256:
        actual_hash = sha256sum(destination)
        if actual_hash != remote.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {destination}: expected {remote.sha256}, found {actual_hash}"
            )

    if destination.suffix in ARCHIVE_EXTENSIONS:
        for member in iter_members(destination):
            LOGGER.debug("Extracted %s", safe_relative_path(member))

    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir("raw", "bbbc021"),
        help="Directory where the BBBC021 subset will be stored",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    root = args.output
    ensure_dir(root)

    LOGGER.info("Saving BBBC021 subset to %s", safe_relative_path(root))
    for remote in MANIFEST:
        try:
            download_file(remote, root)
        except Exception as exc:  # pragma: no cover - network edge cases
            LOGGER.error("Failed to download %s: %s", remote.url, exc)
            return 1

    LOGGER.info("Finished downloading %d files", len(MANIFEST))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
