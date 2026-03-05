"""Unit tests for upp.stages.reweight — per-reader jet capping and StopIteration handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import yaml

from upp.stages.reweight import Reweight


def _make_organised_components(tmpdir, jets_per_flavour):
    """Write a minimal organised-components.yaml and return its path.

    Parameters
    ----------
    tmpdir : Path
        Directory to write files into.
    jets_per_flavour : dict[str, int]
        Mapping of flavour name to jet count (used to create mock H5 files).

    Returns
    -------
    Path
        Path to the organised-components.yaml file.
    """
    files = {}
    for flav, n in jets_per_flavour.items():
        fpath = tmpdir / f"{flav}.h5"
        # Create a minimal HDF5 file with n jets
        import h5py

        with h5py.File(fpath, "w") as f:
            dtype = np.dtype([("pt", "f4"), ("eta", "f4"), ("flavour_label", "i4")])
            data = np.zeros(n, dtype=dtype)
            f.create_dataset("jets", data=data)
        files[flav] = [str(fpath)]

    config = {"files": {"train": files}, "num_jets": {"train": jets_per_flavour}}
    config_path = tmpdir / "organised-components.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def _make_reweight_obj(tmpdir, jets_per_flavour, num_jets_estimate, batch_size=1000):
    """Create a Reweight instance with mocked config."""
    config_path = _make_organised_components(tmpdir, jets_per_flavour)

    config = MagicMock()
    config.batch_size = batch_size
    config.base_dir = str(tmpdir)

    rw_config = SimpleNamespace(num_jets_estimate=num_jets_estimate, reweights=[])

    rw = object.__new__(Reweight)
    rw.config = config
    rw.rw_config = rw_config
    rw.flavours = list(jets_per_flavour.keys())
    rw.organised_components_config = config_path
    return rw


class TestGetInputReaders:
    def test_caps_at_available_jets(self, tmp_path):
        """When a reader has fewer jets than num_jets_estimate, cap to available."""
        rw = _make_reweight_obj(
            tmp_path,
            jets_per_flavour={"bjets": 50, "cjets": 200},
            num_jets_estimate=100,
        )
        readers, per_reader_num_jets = rw.get_input_readers()
        assert len(readers) == 2
        assert len(per_reader_num_jets) == 2
        # bjets has 50 < 100, should be capped at 50
        assert per_reader_num_jets[0] == 50
        # cjets has 200 >= 100, should use estimate
        assert per_reader_num_jets[1] == 100

    def test_all_above_estimate(self, tmp_path):
        """When all readers have enough jets, use num_jets_estimate for all."""
        rw = _make_reweight_obj(
            tmp_path,
            jets_per_flavour={"bjets": 500, "cjets": 300},
            num_jets_estimate=100,
        )
        _, per_reader_num_jets = rw.get_input_readers()
        assert per_reader_num_jets == [100, 100]

    def test_warning_printed(self, tmp_path, capsys):
        """Capping prints a warning message."""
        rw = _make_reweight_obj(
            tmp_path,
            jets_per_flavour={"bjets": 30},
            num_jets_estimate=100,
        )
        rw.get_input_readers()
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "30" in captured.out


class TestCalculateWeightsStopIteration:
    """Test that the batch loop handles StopIteration from shorter readers."""

    def test_unequal_reader_lengths(self, tmp_path):
        """Readers with different num_jets don't crash the batch loop."""
        rw = _make_reweight_obj(
            tmp_path,
            jets_per_flavour={"bjets": 50, "cjets": 200},
            num_jets_estimate=200,
            batch_size=100,
        )

        # Create a minimal reweight config
        rw_spec = SimpleNamespace(
            group="jets",
            reweight_vars=["pt"],
            flat_bins=[np.array([0, 1, 2], dtype="f4")],
            class_var="flavour_label",
            class_target="mean",
            target_hist_func=None,
        )
        rw_spec.__repr__ = lambda _self: "test_rw"
        rw.rw_config.reweights = [rw_spec]

        # Run calculate_weights — should not raise StopIteration
        result = rw.calculate_weights()
        assert "jets" in result
