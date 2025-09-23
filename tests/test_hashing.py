"""Tests for hashing module."""

import json
import tempfile
from pathlib import Path

import pytest

from cp_tox_mini.hashing import (
    generate_manifest_for_directory,
    load_manifest,
    record_manifest,
    sha256_bytes,
    sha256_file,
    validate_manifest,
)


def test_sha256_bytes():
    """Test SHA256 computation for bytes."""
    # Known SHA256 values
    assert sha256_bytes(b"hello") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    
    # Test deterministic behavior
    data = b"test data for reproducibility"
    hash1 = sha256_bytes(data)
    hash2 = sha256_bytes(data)
    assert hash1 == hash2


def test_sha256_file():
    """Test SHA256 computation for files."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write("hello world")
        tmp_path = Path(tmp.name)
    
    try:
        # Known SHA256 for "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert sha256_file(tmp_path) == expected
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            sha256_file("nonexistent_file.txt")
            
    finally:
        tmp_path.unlink()


def test_manifest_roundtrip():
    """Test manifest creation, loading, and validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test files
        test_files = ["file1.txt", "file2.txt", "subdir/file3.txt"]
        test_data = ["content 1", "content 2", "content 3"]
        
        for filename, content in zip(test_files, test_data):
            file_path = tmpdir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        
        # Generate manifest entries
        entries = generate_manifest_for_directory(tmpdir, "*", tmpdir)
        
        assert len(entries) == 3
        assert all("relpath" in entry for entry in entries)
        assert all("sha256" in entry for entry in entries)
        assert all("size" in entry for entry in entries)
        
        # Check sorting
        relpaths = [entry["relpath"] for entry in entries]
        assert relpaths == sorted(relpaths)
        
        # Save manifest
        manifest_path = tmpdir / "manifest.json"
        record_manifest(entries, manifest_path)
        
        # Load and validate
        manifest = load_manifest(manifest_path)
        assert manifest["version"] == 1
        assert "generated_at_utc" in manifest
        assert len(manifest["files"]) == 3
        
        # Validate files
        assert validate_manifest(manifest_path, tmpdir)
        
        # Test validation failure - modify a file
        (tmpdir / "file1.txt").write_text("modified content")
        assert not validate_manifest(manifest_path, tmpdir)


def test_manifest_schema_validation():
    """Test manifest schema validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test invalid manifest - missing files key
        invalid_manifest = {"version": 1}
        manifest_path = tmpdir / "invalid.json"
        with open(manifest_path, "w") as f:
            json.dump(invalid_manifest, f)
        
        with pytest.raises(ValueError, match="missing 'files' key"):
            validate_manifest(manifest_path, tmpdir)
        
        # Test invalid entry - missing required keys
        invalid_entries = [{"relpath": "file.txt"}]  # missing sha256, size
        invalid_manifest = {"version": 1, "files": invalid_entries}
        
        with open(manifest_path, "w") as f:
            json.dump(invalid_manifest, f)
        
        with pytest.raises(ValueError, match="missing required keys"):
            validate_manifest(manifest_path, tmpdir)


def test_deterministic_hashing():
    """Test that hashing is deterministic across runs."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write("deterministic test content")
        tmp_path = Path(tmp.name)
    
    try:
        # Multiple hash computations should give same result
        hashes = [sha256_file(tmp_path) for _ in range(5)]
        assert all(h == hashes[0] for h in hashes)
        
    finally:
        tmp_path.unlink()


def test_empty_directory_manifest():
    """Test manifest generation for empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        entries = generate_manifest_for_directory(tmpdir, "*", tmpdir)
        assert entries == []


def test_manifest_file_not_found():
    """Test behavior when manifest file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_manifest("nonexistent_manifest.json")
    
    with pytest.raises(FileNotFoundError):
        validate_manifest("nonexistent_manifest.json")


def test_relative_path_handling():
    """Test that relative paths are computed correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create nested directory structure
        data_dir = tmpdir / "data" / "processed"
        data_dir.mkdir(parents=True)
        
        test_file = data_dir / "test.txt"
        test_file.write_text("test content")
        
        # Generate manifest with different base directories
        entries_from_tmpdir = generate_manifest_for_directory(data_dir, "*", tmpdir)
        entries_from_data = generate_manifest_for_directory(data_dir, "*", data_dir)
        
        assert len(entries_from_tmpdir) == 1
        assert len(entries_from_data) == 1
        
        # Check relative paths
        assert entries_from_tmpdir[0]["relpath"] == "data/processed/test.txt"
        assert entries_from_data[0]["relpath"] == "test.txt"