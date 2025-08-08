"""Unit tests for the Reweight class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.stages.reweight import Reweight


class TestReweight:
    """Test class for the Reweight stage."""

    @pytest.fixture
    def test_config_path(self):
        """Get path to test config file."""
        return Path(__file__).parent.parent / "fixtures" / "test_config_rw.yaml"

    @pytest.fixture
    def organised_components_file(self, tmp_path):
        """Create a mock organised components YAML file."""
        components_data = {
            "files": {
                "train": {
                    "bjets": [str(tmp_path / "bjets1.h5"), str(tmp_path / "bjets2.h5")],
                    "cjets": [str(tmp_path / "cjets1.h5"), str(tmp_path / "cjets2.h5")],
                    "ujets": [str(tmp_path / "ujets1.h5"), str(tmp_path / "ujets2.h5")]
                }
            },
            "num_jets": {
                "train": {
                    "bjets": 10000,
                    "cjets": 8000,
                    "ujets": 12000
                }
            }
        }

        components_dir = tmp_path / "split-components"
        components_dir.mkdir()
        components_file = components_dir / "organised-components.yaml"

        with open(components_file, "w") as f:
            yaml.dump(components_data, f)

        return components_file

    @pytest.fixture
    def mock_config(self, test_config_path, organised_components_file, tmp_path):
        """Create a real preprocessing config from test fixture."""
        config = PreprocessingConfig.from_file(
            test_config_path,
            "train",
            skip_checks=True,
            skip_config_copy=True
        )
        # Override paths to use our temp directory
        config.base_dir = organised_components_file.parent.parent
        config.out_dir = tmp_path / "test_out"
        return config

    def test_init_success(self, mock_config):
        """Test successful initialization of Reweight class."""
        reweight = Reweight(mock_config)

        assert reweight.config == mock_config
        assert reweight.rw_config == mock_config.rw_config
        assert isinstance(reweight.flavours, list)
        assert len(reweight.flavours) > 0

    def test_init_no_rw_config(self, test_config_path, organised_components_file):
        """Test initialization fails when reweight config is None."""
        config = PreprocessingConfig.from_file(
            test_config_path,
            "train",
            skip_checks=True,
            skip_config_copy=True
        )
        config.base_dir = organised_components_file.parent.parent
        config.rw_config = None

        with pytest.raises(AssertionError, match="Reweighting configuration is not set"):
            Reweight(config)

    def test_init_no_organised_components(self, test_config_path):
        """Test initialization fails when organised components file doesn't exist."""
        config = PreprocessingConfig.from_file(
            test_config_path,
            "train",
            skip_checks=True,
            skip_config_copy=True
        )
        config.base_dir = Path("/nonexistent")

        with pytest.raises(AssertionError, match="Organised components config file not found"):
            Reweight(config)

    def test_hists_path_property(self, mock_config):
        """Test hists_path property returns correct path."""
        reweight = Reweight(mock_config)
        expected_path = mock_config.out_dir / "histograms.h5"
        assert reweight.hists_path == expected_path

    def test_num_jets_estimate_property(self, mock_config):
        """Test num_jets_estimate uses reweight config value when available."""
        reweight = Reweight(mock_config)
        # Should use the value from the reweight config or fall back to main config
        assert isinstance(reweight.num_jets_estimate, int)
        assert reweight.num_jets_estimate > 0

    def test_save_and_load_weights_hdf5(self, tmp_path):
        """Test saving and loading weights to/from HDF5."""
        weights_dict = {
            "jets": {
                "weight_test": {
                    "bins": [
                        np.array([0, 1, 2, 3]),
                        np.array([-1, 0, 1])
                    ],
                    "weights": {
                        "0": np.array([[1.0, 2.0], [3.0, 4.0]]),
                        "1": np.array([[0.5, 1.5], [2.5, 3.5]])
                    },
                    "rw_vars": ["pt", "eta"],
                    "class_var": "flavour_label"
                }
            }
        }

        weights_file = tmp_path / "test_weights.h5"

        # Test saving
        Reweight.save_weights_hdf5(weights_dict, weights_file)
        assert weights_file.exists()

        # Test loading
        loaded_weights = Reweight.load_weights_hdf5(weights_file)

        # Verify structure
        assert "jets" in loaded_weights
        assert "weight_test" in loaded_weights["jets"]

        loaded_data = loaded_weights["jets"]["weight_test"]
        assert "bins" in loaded_data
        assert "weights" in loaded_data
        assert "rw_vars" in loaded_data
        assert "class_var" in loaded_data

        # Verify data integrity
        assert len(loaded_data["bins"]) == 2
        np.testing.assert_array_equal(
            loaded_data["bins"][0],
            weights_dict["jets"]["weight_test"]["bins"][0]
        )
        np.testing.assert_array_equal(
            loaded_data["bins"][1],
            weights_dict["jets"]["weight_test"]["bins"][1]
        )

        assert loaded_data["rw_vars"] == ["pt", "eta"]
        assert loaded_data["class_var"] == "flavour_label"

        np.testing.assert_array_equal(
            loaded_data["weights"]["0"],
            weights_dict["jets"]["weight_test"]["weights"]["0"]
        )
        np.testing.assert_array_equal(
            loaded_data["weights"]["1"],
            weights_dict["jets"]["weight_test"]["weights"]["1"]
        )

    @patch('upp.stages.reweight.Reweight.calculate_weights')
    @patch('upp.stages.reweight.Reweight.save_weights_hdf5')
    @patch('upp.stages.reweight.Reweight.plot_rw_histograms')
    def test_run(self, mock_plot, mock_save, mock_calculate, mock_config):
        """Test the main run method."""
        # Mock return values
        mock_weights = {"jets": {"weight_test": {}}}
        mock_calculate.return_value = mock_weights

        reweight = Reweight(mock_config)
        reweight.run()

        # Verify all methods were called
        mock_calculate.assert_called_once()
        mock_save.assert_called_once_with(mock_weights, reweight.hists_path)
        mock_plot.assert_called_once_with(reweight.hists_path)

    def test_save_weights_hdf5_creates_directory(self, tmp_path):
        """Test that save_weights_hdf5 creates parent directories."""
        weights_dict = {
            "jets": {
                "weight_test": {
                    "bins": [np.array([0, 1, 2])],
                    "weights": {"0": np.array([1.0, 2.0])},
                    "rw_vars": ["pt"],
                    "class_var": "flavour_label"
                }
            }
        }

        # Use a nested path that doesn't exist
        weights_file = tmp_path / "nested" / "dir" / "weights.h5"

        Reweight.save_weights_hdf5(weights_dict, weights_file)

        assert weights_file.exists()
        assert weights_file.parent.exists()

    def test_load_weights_hdf5_nonexistent_file(self, tmp_path):
        """Test load_weights_hdf5 with nonexistent file."""
        nonexistent_file = tmp_path / "nonexistent.h5"

        with pytest.raises(FileNotFoundError):
            Reweight.load_weights_hdf5(nonexistent_file)
