"""Unit tests for upp.stages.split_containers."""

from __future__ import annotations

from pathlib import Path

import h5py

from upp.stages.split_containers import validate_h5_file


def _write_valid_h5(path: Path) -> None:
    """Create a minimal valid HDF5 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("jets", shape=(10,), dtype="f4")


def _write_corrupt_h5(path: Path) -> None:
    """Create a corrupt (non-HDF5) file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"NOT AN HDF5 FILE - CORRUPTED DATA")


class TestValidateH5File:
    """Tests for validate_h5_file function."""

    def test_valid_file_returns_true(self, tmp_path: Path) -> None:
        """A valid HDF5 file should return True."""
        valid_file = tmp_path / "valid.h5"
        _write_valid_h5(valid_file)
        assert validate_h5_file(valid_file) is True

    def test_corrupt_file_returns_false(self, tmp_path: Path) -> None:
        """A corrupt file should return False."""
        corrupt_file = tmp_path / "corrupt.h5"
        _write_corrupt_h5(corrupt_file)
        assert validate_h5_file(corrupt_file) is False

    def test_nonexistent_file_returns_false(self, tmp_path: Path) -> None:
        """A non-existent file should return False."""
        missing_file = tmp_path / "missing.h5"
        assert validate_h5_file(missing_file) is False

    def test_empty_file_returns_false(self, tmp_path: Path) -> None:
        """An empty file should return False."""
        empty_file = tmp_path / "empty.h5"
        empty_file.touch()
        assert validate_h5_file(empty_file) is False

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Function should accept string paths."""
        valid_file = tmp_path / "valid.h5"
        _write_valid_h5(valid_file)
        assert validate_h5_file(str(valid_file)) is True
