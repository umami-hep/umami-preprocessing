# tests/test_preprocessing_config_unittest.py
from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from dotmap import DotMap
from ftag import Extended_Flavours, Flavours, LabelContainer, get_mock_file

from upp import __version__
from upp.classes.preprocessing_config import PreprocessingConfig


class TestPreprocessingConfig(unittest.TestCase):
    """unittest-based rewrite of the original pytest suite."""

    # Path to the fixtures directory
    CFG_DIR = Path(__file__).parent.parent / "fixtures"

    # ---------- helpers ----------

    @staticmethod
    def generate_mock(out_file: str | Path, N: int = 10) -> None:
        """Create a small mock ntuple file."""
        _, f = get_mock_file(num_jets=N, fname=str(out_file))
        f.close()

    # ---------- test scaffolding ----------

    def setUp(self) -> None:
        """Create the data mock files for the PreprocessingConfig testcase."""
        os.makedirs("/tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.generate_mock("/tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5")
        self.generate_mock("/tmp/upp-tests/integration/temp_workspace/ntuples/data2.h5")

    def tearDown(self) -> None:
        """Clean up after each test."""
        subprocess.run(["rm", "-rf", "/tmp/upp-tests/integration"], check=True)

    # ---------- individual tests ----------

    def test_get_umami_general(self) -> None:
        fpath = self.CFG_DIR / "test_config_pdf_auto_umami.yaml"
        config = PreprocessingConfig.from_file(fpath, "train")
        general = config.get_umami_general()
        self.assertEqual(general["dict_file"], "dict/file/path.json")

    def test_legacy_plotting_jet_count(self) -> None:
        config = PreprocessingConfig.from_file(
            self.CFG_DIR / "test_config_pdf_auto_umami.yaml",
            "train",
            skip_checks=True,
            skip_config_copy=True,
        )

        self.assertEqual(config.plotting.num_jets_plotting, config.num_jets_estimate_plotting)

    def test_plotting_config(self) -> None:
        config = PreprocessingConfig.from_file(
            self.CFG_DIR.parent.parent / "integration/fixtures/test_config_pdf_auto.yaml",
            "train",
            skip_checks=True,
            skip_config_copy=True,
        )

        self.assertEqual(config.plotting.num_jets_plotting, 100)
        self.assertEqual(config.plotting.variable_label("pt"), "$p_\\mathrm{T}$ [GeV]")
        self.assertEqual(config.plotting.variable_label("mass"), "Jet Mass [GeV]")
        self.assertEqual(config.plotting.sample_label("ttbar"), "$t\\bar{t}$")
        self.assertEqual(config.plotting.output_formats, ["png"])

    def test_mimic_umami_config(self) -> None:
        config = PreprocessingConfig.from_file(
            self.CFG_DIR / "test_config_pdf_auto_umami.yaml",
            "train",
        )
        general = DotMap(config.get_umami_general(), _dynamic=False)
        config.mimic_umami_config(general)
        self.assertEqual(config.general.dict_file, "dict/file/path.json")

    def test_mimic_umami_config_required(self) -> None:
        config = PreprocessingConfig.from_file(
            self.CFG_DIR / "test_config_pdf_auto_umami_required.yaml",
            "train",
        )
        general = DotMap(config.get_umami_general(), _dynamic=False)
        config.mimic_umami_config(general)
        self.assertEqual(config.general.dict_file, "dict/file/path.json")

    def test_get_file_name(self) -> None:
        config = PreprocessingConfig.from_file(
            self.CFG_DIR / "test_config_pdf_auto_umami.yaml",
            "train",
        )

        # Valid cases
        self.assertEqual(
            str(config.get_file_name("resampled")),
            "/tmp/upp-tests/integration/temp_workspace/test_out/pp_output_train.h5",
        )
        self.assertEqual(
            str(config.get_file_name("resampled_scaled_shuffled")),
            "/tmp/upp-tests/integration/temp_workspace/"
            "test_out/pp_output_train_resampled_scaled_shuffled.h5",
        )

        # Invalid case
        with self.assertRaises(ValueError):
            config.get_file_name("invalid_stage")

    def test_not_existing_dir(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            PreprocessingConfig(
                config_path=Path("/tmp/upp-tests/integration/temp_workspace/test.yaml"),
                split="train",
                config={
                    "resampling": {"variables": {"jets": {"labels": ["test"]}}, "target": "bjets"},
                    "components": [],
                    "variables": {"jets": {"labels": ["test"]}},
                },
                base_dir=Path("/tmp/error/"),
            )

        self.assertEqual("Path /tmp/error/ntuples does not exist", str(ctx.exception))

    def test_git_hash_missing_git(self) -> None:
        # git not installed -> get_git_hash raises FileNotFoundError; fall back to version
        with patch(
            "upp.classes.preprocessing_config.get_git_hash",
            side_effect=FileNotFoundError,
        ):
            config = PreprocessingConfig(
                config_path=Path("/tmp/upp-tests/integration/temp_workspace/test.yaml"),
                split="train",
                config={
                    "resampling": {"variables": {"jets": {"labels": ["test"]}}, "target": "bjets"},
                    "components": [],
                    "variables": {"jets": {"labels": ["test"]}},
                },
                base_dir=Path("/tmp/upp-tests/integration/temp_workspace/"),
            )
        self.assertEqual(config.git_hash, __version__)
        self.assertEqual(config.config["upp_hash"], __version__)

    def test_standard_flavour_config(self) -> None:
        config = PreprocessingConfig(
            config_path=Path("/tmp/upp-tests/integration/temp_workspace/test.yaml"),
            split="train",
            config={
                "resampling": {"variables": {"jets": {"labels": ["test"]}}, "target": "bjets"},
                "components": [],
                "variables": {"jets": {"labels": ["test"]}},
            },
            base_dir=Path("/tmp/upp-tests/integration/temp_workspace/"),
            flavour_category="standard",
        )
        self.assertEqual(config.flavour_cont, Flavours)

    def test_extended_flavour_config(self) -> None:
        config = PreprocessingConfig(
            config_path=Path("/tmp/upp-tests/integration/temp_workspace/test.yaml"),
            split="train",
            config={
                "resampling": {"variables": {"jets": {"labels": ["test"]}}, "target": "bjets"},
                "components": [],
                "variables": {"jets": {"labels": ["test"]}},
            },
            base_dir=Path("/tmp/upp-tests/integration/temp_workspace/"),
            flavour_category="extended",
        )
        self.assertEqual(config.flavour_cont, Extended_Flavours)

    def test_unsupported_flavour_config(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PreprocessingConfig(
                config_path=Path("/tmp/upp-tests/integration/temp_workspace/test.yaml"),
                split="train",
                config={
                    "resampling": {"variables": {"jets": {"labels": ["test"]}}, "target": "bjets"},
                    "components": [],
                    "variables": {"jets": {"labels": ["test"]}},
                },
                base_dir=Path("/tmp/upp-tests/integration/temp_workspace/"),
                flavour_category="error",
            )

        self.assertEqual(
            "flavour_category error is not supported in the default "
            + "flavours! If you want to use your own flavour config yaml file, please "
            + "provide flavour_config!",
            str(ctx.exception),
        )

    def test_separate_flavour_config(self) -> None:
        config = PreprocessingConfig(
            config_path=Path("/tmp/upp-tests/integration/temp_workspace/test.yaml"),
            split="train",
            config={
                "resampling": {"variables": {"jets": {"labels": ["test"]}}, "target": "bjets"},
                "components": [],
                "variables": {"jets": {"labels": ["test"]}},
            },
            base_dir=Path("/tmp/upp-tests/integration/temp_workspace/"),
            flavour_config=self.CFG_DIR / "test_flavour_config.yaml",
        )
        self.assertEqual(
            config.flavour_cont,
            LabelContainer.from_yaml(yaml_path=self.CFG_DIR / "test_flavour_config.yaml"),
        )
