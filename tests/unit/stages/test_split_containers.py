"""Unit tests for upp.stages.split_containers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import pytest

from upp.stages.split_containers import SplitContainers, validate_h5_file


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


class TestMakeTmpVds:
    """Tests for _make_tmp_vds method."""

    @pytest.fixture
    def mock_split_containers(self, tmp_path: Path) -> SplitContainers:
        """Create a SplitContainers instance with mocked config."""
        with patch.object(SplitContainers, "__init__", lambda _self, _x: None):
            sc = SplitContainers(None)
            sc.config_path = tmp_path / "config.yaml"
            sc.config = SimpleNamespace(
                base_dir=str(tmp_path),
                ntuple_dir=str(tmp_path / "ntuples"),
                batch_size=1000,
            )
            return sc

    def test_single_file_path_yields_directly(
        self, mock_split_containers: SplitContainers, tmp_path: Path
    ) -> None:
        """When a single file path is passed, it should yield directly."""
        single_file = tmp_path / "single.h5"
        _write_valid_h5(single_file)

        with mock_split_containers._make_tmp_vds(single_file) as result:
            assert result == single_file

    def test_single_file_string_yields_directly(
        self, mock_split_containers: SplitContainers, tmp_path: Path
    ) -> None:
        """When a single file string is passed, it should yield directly."""
        single_file = tmp_path / "single.h5"
        _write_valid_h5(single_file)

        with mock_split_containers._make_tmp_vds(str(single_file)) as result:
            assert result == single_file

    def test_all_corrupted_files_raises_error(
        self, mock_split_containers: SplitContainers, tmp_path: Path
    ) -> None:
        """When all files are corrupted, should raise RuntimeError."""
        corrupt1 = tmp_path / "corrupt1.h5"
        corrupt2 = tmp_path / "corrupt2.h5"
        _write_corrupt_h5(corrupt1)
        _write_corrupt_h5(corrupt2)

        with (
            pytest.raises(RuntimeError, match="No valid HDF5 files found"),
            mock_split_containers._make_tmp_vds([str(corrupt1), str(corrupt2)]),
        ):
            pass

    def test_mixed_files_filters_corrupted(
        self,
        mock_split_containers: SplitContainers,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When some files are corrupted, they should be filtered out with warning."""
        valid1 = tmp_path / "valid1.h5"
        valid2 = tmp_path / "valid2.h5"
        corrupt = tmp_path / "corrupt.h5"
        _write_valid_h5(valid1)
        _write_valid_h5(valid2)
        _write_corrupt_h5(corrupt)

        # Mock create_virtual_file and H5Reader to avoid actual VDS creation
        mock_reader = MagicMock()
        mock_reader.num_jets = 20

        with (
            patch("upp.stages.split_containers.create_virtual_file"),
            patch("upp.stages.split_containers.H5Reader", return_value=mock_reader),
            mock_split_containers._make_tmp_vds([str(valid1), str(valid2), str(corrupt)]) as result,
        ):
            assert result.name == "combined.h5"

        # Check that warnings were logged
        assert "Skipping corrupted file" in caplog.text
        assert "Skipped 1/3 corrupted files" in caplog.text

    def test_all_valid_files_creates_vds(
        self,
        mock_split_containers: SplitContainers,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When all files are valid, VDS should be created without warnings."""
        valid1 = tmp_path / "valid1.h5"
        valid2 = tmp_path / "valid2.h5"
        _write_valid_h5(valid1)
        _write_valid_h5(valid2)

        mock_reader = MagicMock()
        mock_reader.num_jets = 20

        with (
            patch("upp.stages.split_containers.create_virtual_file"),
            patch("upp.stages.split_containers.H5Reader", return_value=mock_reader),
            mock_split_containers._make_tmp_vds([str(valid1), str(valid2)]) as result,
        ):
            assert result.name == "combined.h5"

        # No skip warnings should be logged
        assert "Skipped" not in caplog.text

    def test_symlinks_created_for_valid_files(
        self, mock_split_containers: SplitContainers, tmp_path: Path
    ) -> None:
        """Valid files should have symlinks created in temp directory."""
        valid1 = tmp_path / "valid1.h5"
        valid2 = tmp_path / "valid2.h5"
        corrupt = tmp_path / "corrupt.h5"
        _write_valid_h5(valid1)
        _write_valid_h5(valid2)
        _write_corrupt_h5(corrupt)

        created_symlinks = []

        def capture_vds_call(pattern, *_args, **_kwargs):
            # Capture the directory from the pattern to check symlinks
            import glob as glob_mod

            matches = glob_mod.glob(pattern)
            created_symlinks.extend(matches)

        mock_reader = MagicMock()
        mock_reader.num_jets = 20

        with (
            patch("upp.stages.split_containers.create_virtual_file", side_effect=capture_vds_call),
            patch("upp.stages.split_containers.H5Reader", return_value=mock_reader),
            mock_split_containers._make_tmp_vds([str(valid1), str(valid2), str(corrupt)]),
        ):
            # Check that only valid files were symlinked
            assert len(created_symlinks) == 2
            symlink_names = [Path(s).name for s in created_symlinks]
            assert "valid1.h5" in symlink_names
            assert "valid2.h5" in symlink_names
            assert "corrupt.h5" not in symlink_names
