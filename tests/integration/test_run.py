from __future__ import annotations

import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
from ftag import get_mock_file

from upp.main import main

this_dir = Path(__file__).parent


class TestClass:
    def generate_mock(self, out_file, N=100_000):
        _, f = get_mock_file(num_jets=N, fname=out_file)
        f.close()

    def setup_method(self, method):
        os.makedirs("tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5")
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data2.h5")
        print(f"setup_method, method: {method.__name__}")

    def teardown_method(self, method):
        subprocess.run(["rm", "-r", "tmp"], check=True)
        print(f"teardown_method, method: {method.__name__}")

    def test_run_prep_lowpt_ttbar_bjets(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_countup.yaml")),
            "--no-resample",
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--split",
            "train",
            "--component",
            "lowpt_ttbar_ujets",
        ]
        main(args)

    def test_run_prep_error(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_countup.yaml")),
            "--no-resample",
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--split",
            "train",
            "--component",
            "fail",
        ]

        with pytest.raises(ValueError):
            main(args)

    def test_run_pdf_auto(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_pdf_auto.yaml")),
            "--split",
            "train",
        ]
        main(args)
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_pdf_auto.yaml")),
            "--split",
            "val",
        ]
        main(args)
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_pdf_auto.yaml")),
            "--split",
            "test",
        ]
        main(args)

    def test_run_pdf_upscale(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_pdf_upscaled.yaml")),
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--split",
            "train",
        ]
        main(args)

    def test_run_countup(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_countup.yaml")),
            "--split",
            "train",
        ]
        main(args)

    def test_run_track_selector(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_track_selection.yaml")),
            "--no-plot",
            "--split",
            "train",
        ]
        main(args)

        fname = "tmp/upp-tests/integration/temp_workspace/test_out/pp_output_train.h5"
        assert os.path.exists(fname)
        with h5py.File(fname, "r") as f:
            tracks = f["tracks"][:]
        assert np.all(tracks[tracks["valid"]]["d0"] < 3.5)

    def test_run_countup_region_lowpt(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_countup.yaml")),
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--split",
            "train",
            "--region",
            "lowpt",
        ]
        main(args)

    def test_run_countup_component_lowpt_ttbar_cjets(self):
        prep_args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_countup.yaml")),
            "--no-resample",
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--split",
            "train",
            "--component",
            "lowpt_ttbar_bjets",
        ]
        main(prep_args)
        resample_args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_countup.yaml")),
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--split",
            "train",
            "--region",
            "lowpt",
            "--component",
            "lowpt_ttbar_cjets",
        ]
        main(resample_args)

    def test_run_countup_component_error(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_countup.yaml")),
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--split",
            "train",
            "--component",
            "lowpt_ttbar_cjets",
        ]

        with pytest.raises(ValueError):
            main(args)

    def test_run_countup_region_error(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_countup.yaml")),
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--split",
            "train",
            "--region",
            "fail",
        ]

        with pytest.raises(ValueError):
            main(args)

    def test_run_countup_upscaled_error(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_countup_upscaled.yaml")),
            "--no-plot",
            "--split",
            "train",
        ]

        with pytest.raises(ValueError):
            main(args)
