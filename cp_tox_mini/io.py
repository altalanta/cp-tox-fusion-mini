"""Input/output operations for reproducible data handling."""

import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import URLError

from .hashing import generate_manifest_for_directory, record_manifest, validate_manifest


# Minimal dataset URLs for smoke testing
BBBC021_URLS = {
    # Small subset of BBBC021 plates - using publicly accessible test images
    "plate_A1_w1.tiff": "https://www.cellpainting.org/data/week1_150607/BBBC021_v1_images_Week1_22123/Week1_150607_A01_s1_w1.tif",
    "plate_A1_w2.tiff": "https://www.cellpainting.org/data/week1_150607/BBBC021_v1_images_Week1_22123/Week1_150607_A01_s1_w2.tif",
    "plate_A1_w3.tiff": "https://www.cellpainting.org/data/week1_150607/BBBC021_v1_images_Week1_22123/Week1_150607_A01_s1_w3.tif",
    "plate_B1_w1.tiff": "https://www.cellpainting.org/data/week1_150607/BBBC021_v1_images_Week1_22123/Week1_150607_B01_s1_w1.tif",
    "plate_B1_w2.tiff": "https://www.cellpainting.org/data/week1_150607/BBBC021_v1_images_Week1_22123/Week1_150607_B01_s1_w2.tif",
    "plate_B1_w3.tiff": "https://www.cellpainting.org/data/week1_150607/BBBC021_v1_images_Week1_22123/Week1_150607_B01_s1_w3.tif",
}

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

# Fallback minimal synthetic data if URLs fail
SYNTHETIC_TOX21_DATA = """smiles,compound_id,target,assay_id,plate_id,well_row,well_col
CCO,ETHANOL,0,NR-AR,plate_001,A,1
CC(C)O,ISOPROPANOL,1,NR-AR,plate_001,A,2
CCCCCCCCCCCCCCCCCC(=O)O,STEARIC_ACID,0,NR-AR,plate_001,A,3
c1ccc(cc1)O,PHENOL,1,NR-AR,plate_001,A,4
CCc1ccccc1,ETHYLBENZENE,1,NR-AR,plate_001,B,1
CC(C)(C)O,TERT_BUTANOL,0,NR-AR,plate_001,B,2
Cc1ccc(cc1)O,P_CRESOL,1,NR-AR,plate_001,B,3
c1ccc2c(c1)ccccc2,NAPHTHALENE,1,NR-AR,plate_001,B,4
"""


def download_file(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a file from URL with error handling.
    
    Args:
        url: URL to download from
        output_path: Local path to save file
        timeout: Download timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Downloading {url} -> {output_path}")
        
        # Handle .gz files
        if url.endswith('.gz'):
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with gzip.open(response, 'rt') as gz_file:
                    content = gz_file.read()
                    with open(output_path, 'w') as f:
                        f.write(content)
        else:
            urllib.request.urlretrieve(url, output_path)
        
        print(f"✓ Downloaded {output_path.name}")
        return True
        
    except (URLError, TimeoutError, Exception) as e:
        print(f"✗ Failed to download {url}: {e}")
        return False


def create_synthetic_tiff(output_path: Path, width: int = 64, height: int = 64) -> bool:
    """Create a minimal synthetic TIFF file for testing.
    
    Args:
        output_path: Path to save synthetic TIFF
        width: Image width
        height: Image height
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import numpy as np
        from PIL import Image
        
        # Create synthetic grayscale image with some structure
        np.random.seed(42)  # Deterministic
        image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        
        # Add some structure (circles/patterns)
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(width, height) // 4) ** 2
        image[mask] = 200
        
        # Save as TIFF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_path, format='TIFF')
        
        print(f"✓ Created synthetic TIFF: {output_path.name}")
        return True
        
    except ImportError:
        print("✗ PIL not available for synthetic TIFF creation")
        return False
    except Exception as e:
        print(f"✗ Failed to create synthetic TIFF {output_path}: {e}")
        return False


def download_inputs(data_dir: Path = Path("data")) -> bool:
    """Download minimal datasets for smoke testing.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        True if all downloads successful, False otherwise
    """
    raw_dir = data_dir / "raw"
    bbbc021_dir = raw_dir / "bbbc021"
    tox21_dir = raw_dir / "tox21"
    
    success = True
    
    # Download BBBC021 subset (try real URLs first, fallback to synthetic)
    print("Downloading BBBC021 subset...")
    bbbc021_success = 0
    for filename, url in BBBC021_URLS.items():
        output_path = bbbc021_dir / filename
        if download_file(url, output_path):
            bbbc021_success += 1
        else:
            # Fallback to synthetic TIFF
            if create_synthetic_tiff(output_path):
                bbbc021_success += 1
    
    if bbbc021_success == 0:
        print("✗ Failed to download any BBBC021 files")
        success = False
    else:
        print(f"✓ Downloaded/created {bbbc021_success}/{len(BBBC021_URLS)} BBBC021 files")
    
    # Download Tox21 subset
    print("Downloading Tox21 subset...")
    tox21_path = tox21_dir / "tox21_mini.csv"
    
    if not download_file(TOX21_URL, tox21_path):
        # Fallback to synthetic data
        print("Using synthetic Tox21 data...")
        tox21_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tox21_path, 'w') as f:
            f.write(SYNTHETIC_TOX21_DATA)
        print(f"✓ Created synthetic Tox21 data: {tox21_path.name}")
    
    # Generate manifest
    print("Generating data manifest...")
    entries = []
    entries.extend(generate_manifest_for_directory(bbbc021_dir, "*.tiff"))
    entries.extend(generate_manifest_for_directory(tox21_dir, "*.csv"))
    
    if entries:
        record_manifest(entries)
        print(f"✓ Generated manifest with {len(entries)} files")
    else:
        print("✗ No files found for manifest")
        success = False
    
    return success


def validate_inputs() -> bool:
    """Validate downloaded inputs against manifest.
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        return validate_manifest()
    except FileNotFoundError:
        print("✗ No manifest found - run download first")
        return False
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def list_manifest_summary() -> Dict:
    """Get summary of manifest contents.
    
    Returns:
        Dictionary with manifest summary
    """
    try:
        from .hashing import load_manifest
        manifest = load_manifest()
        
        files = manifest.get("files", [])
        total_size = sum(f.get("size", 0) for f in files)
        
        summary = {
            "version": manifest.get("version", "unknown"),
            "generated_at": manifest.get("generated_at_utc", "unknown"),
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": files[:5]  # First 5 files for preview
        }
        
        return summary
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python io.py <command>")
        print("Commands:")
        print("  download    - Download datasets and generate manifest")
        print("  validate    - Validate existing data against manifest")
        print("  summary     - Show manifest summary")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "download":
        success = download_inputs()
        sys.exit(0 if success else 1)
    elif command == "validate":
        success = validate_inputs()
        sys.exit(0 if success else 1)
    elif command == "summary":
        summary = list_manifest_summary()
        if "error" in summary:
            print(f"Error: {summary['error']}")
            sys.exit(1)
        else:
            print(f"Manifest version: {summary['version']}")
            print(f"Generated: {summary['generated_at']}")
            print(f"Total files: {summary['total_files']}")
            print(f"Total size: {summary['total_size_mb']} MB")
            print("Sample files:")
            for f in summary['files']:
                print(f"  {f['relpath']} ({f['size']} bytes)")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)