"""Hash computation and manifest management for reproducible data handling."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union


def sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA256 hash of a file.
    
    Args:
        path: Path to file
        
    Returns:
        Hexadecimal SHA256 hash string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    sha256_hash = hashlib.sha256()
    
    try:
        with open(path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
    except IOError as e:
        raise IOError(f"Cannot read file {path}: {e}")
    
    return sha256_hash.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of byte data.
    
    Args:
        data: Byte data to hash
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    return hashlib.sha256(data).hexdigest()


def record_manifest(entries: List[Dict], manifest_path: Union[str, Path] = "manifests/data_manifest.json") -> None:
    """Record data manifest with file hashes and metadata.
    
    Args:
        entries: List of dicts with keys: relpath, sha256, size
        manifest_path: Path to save manifest JSON
        
    Example:
        entries = [
            {"relpath": "data/raw/plate_A1.tiff", "sha256": "abc123...", "size": 12345},
            {"relpath": "data/raw/tox21_train.csv", "sha256": "def456...", "size": 67890}
        ]
        record_manifest(entries)
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "version": 1,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "files": entries
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_manifest(manifest_path: Union[str, Path] = "manifests/data_manifest.json") -> Dict:
    """Load data manifest from JSON file.
    
    Args:
        manifest_path: Path to manifest JSON
        
    Returns:
        Manifest dictionary
        
    Raises:
        FileNotFoundError: If manifest doesn't exist
        json.JSONDecodeError: If manifest is invalid JSON
    """
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, "r") as f:
        return json.load(f)


def validate_manifest(manifest_path: Union[str, Path] = "manifests/data_manifest.json", 
                     base_dir: Union[str, Path] = ".") -> bool:
    """Validate that files match manifest hashes.
    
    Args:
        manifest_path: Path to manifest JSON
        base_dir: Base directory for resolving relative paths
        
    Returns:
        True if all files match, False otherwise
        
    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest format is invalid
    """
    manifest = load_manifest(manifest_path)
    base_dir = Path(base_dir)
    
    if "files" not in manifest:
        raise ValueError("Manifest missing 'files' key")
    
    all_valid = True
    errors = []
    
    for entry in manifest["files"]:
        if not all(key in entry for key in ["relpath", "sha256", "size"]):
            raise ValueError(f"Manifest entry missing required keys: {entry}")
        
        file_path = base_dir / entry["relpath"]
        
        if not file_path.exists():
            errors.append(f"Missing file: {file_path}")
            all_valid = False
            continue
        
        # Check file size
        actual_size = file_path.stat().st_size
        if actual_size != entry["size"]:
            errors.append(f"Size mismatch {file_path}: expected {entry['size']}, got {actual_size}")
            all_valid = False
            continue
        
        # Check SHA256 hash
        try:
            actual_hash = sha256_file(file_path)
            if actual_hash != entry["sha256"]:
                errors.append(f"Hash mismatch {file_path}: expected {entry['sha256']}, got {actual_hash}")
                all_valid = False
        except (FileNotFoundError, IOError) as e:
            errors.append(f"Cannot hash {file_path}: {e}")
            all_valid = False
    
    if not all_valid:
        print("Manifest validation failed:")
        for error in errors:
            print(f"  - {error}")
    
    return all_valid


def generate_manifest_for_directory(directory: Union[str, Path], 
                                  pattern: str = "*",
                                  base_dir: Union[str, Path] = ".") -> List[Dict]:
    """Generate manifest entries for files in a directory.
    
    Args:
        directory: Directory to scan
        pattern: Glob pattern for files to include
        base_dir: Base directory for computing relative paths
        
    Returns:
        List of manifest entries
    """
    directory = Path(directory)
    base_dir = Path(base_dir)
    entries = []
    
    if not directory.exists():
        return entries
    
    for file_path in directory.rglob(pattern):
        if file_path.is_file():
            try:
                relpath = file_path.relative_to(base_dir)
                size = file_path.stat().st_size
                sha256 = sha256_file(file_path)
                
                entries.append({
                    "relpath": str(relpath),
                    "sha256": sha256,
                    "size": size
                })
            except (ValueError, FileNotFoundError, IOError) as e:
                print(f"Warning: Cannot process {file_path}: {e}")
    
    return sorted(entries, key=lambda x: x["relpath"])


def update_manifest(new_entries: List[Dict], 
                   manifest_path: Union[str, Path] = "manifests/data_manifest.json") -> None:
    """Update existing manifest with new entries.
    
    Args:
        new_entries: New manifest entries to add
        manifest_path: Path to manifest JSON
    """
    try:
        manifest = load_manifest(manifest_path)
        existing_files = {entry["relpath"]: entry for entry in manifest["files"]}
    except FileNotFoundError:
        existing_files = {}
        manifest = {"version": 1, "files": []}
    
    # Update with new entries
    for entry in new_entries:
        existing_files[entry["relpath"]] = entry
    
    # Recreate manifest with updated entries
    updated_entries = list(existing_files.values())
    record_manifest(updated_entries, manifest_path)


if __name__ == "__main__":
    # CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hashing.py <command> [args...]")
        print("Commands:")
        print("  hash <file>              - Compute SHA256 of file")
        print("  generate <dir>           - Generate manifest for directory")
        print("  validate [manifest]      - Validate manifest")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "hash" and len(sys.argv) == 3:
        print(sha256_file(sys.argv[2]))
    elif command == "generate" and len(sys.argv) == 3:
        entries = generate_manifest_for_directory(sys.argv[2])
        for entry in entries:
            print(f"{entry['relpath']}: {entry['sha256']} ({entry['size']} bytes)")
    elif command == "validate":
        manifest_path = sys.argv[2] if len(sys.argv) > 2 else "manifests/data_manifest.json"
        is_valid = validate_manifest(manifest_path)
        print("Manifest validation:", "PASSED" if is_valid else "FAILED")
        sys.exit(0 if is_valid else 1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)