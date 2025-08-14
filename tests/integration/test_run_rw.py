from __future__ import annotations

import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from ftag.mock import JET_VARS, get_mock_file

from upp.main import main

JET_VARS += [("eventNumber", "i4")]

this_dir = Path(__file__).parent


class TestRunRW:
    def generate_mock(self, out_file, N=1_000):
        _, f = get_mock_file(num_jets=N, fname=out_file)
        f["jets"]["eventNumber"] = np.arange(N, dtype="i4")
        f.close()

    def setup_method(self, method):
        os.makedirs("tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5")
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data2.h5")
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data3.h5")
        print(f"setup_method method: {method.__name__}")

    def teardown_method(self, method):
        subprocess.run(["rm", "-r", "tmp"], check=True)
        print(f"teardown_method method: {method.__name__}")

    @property
    def no(self):
        return [
            "--no-resample",
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--no-prep",
        ]

    def _run_split(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_rw.yaml")),
            "--split",
            "train",
            "--split-components",
            *self.no,
        ]
        main(args)
        outpath = Path("tmp/upp-tests/integration/temp_workspace/split-components")

        assert (
            outpath / "organised-components.yaml"
        ).exists(), "Organised components file not found"

        for container in ["data1.h5", "data2.h5", "data3.h5"]:
            assert (outpath / container).exists()
            component = "highpt_zprime" if container == "data3.h5" else "lowpt_ttbar"
            # We expect S * F number of output files, where S is the number of
            # splits and F is the number of flavours
            exp_files = [
                f"output_{split}_{component}_{flavour}.h5"
                for split in ["train", "val", "test"]
                for flavour in ["bjets", "cjets", "ujets", "taujets"]
            ]
            for exp_file in exp_files:
                assert (outpath / container / exp_file).exists(), (
                    f"Expected file {exp_file} not found in {outpath} : "
                    f"Contains {os.listdir(outpath / container)}"
                )

    def _calculate_weights(self):
        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_rw.yaml")),
            "--rw",
            *self.no,
        ]
        main(args)
        outpath = Path("tmp/upp-tests/integration/temp_workspace/test_out")
        hist_file = outpath / "histograms.h5"
        assert hist_file, "Histograms file not found"
        with h5py.File(hist_file, "r") as f:
            print("the stuf?", f.keys())
            assert f.keys() == {
                "jets",
                "tracks",
            }, f"Expected 'jets' key in histograms file only found {f.keys()}"
            jets = f["jets"]
            for dist_target in ["mean", "min", "max", "uniform"]:
                rw_str = f"weight_jets_pt_eta_target_{dist_target}_flavour_label"
                assert rw_str in jets, f"{rw_str}' in jets group"
                w = jets[rw_str]
                assert w.keys() == {"bins", "class_var", "rw_vars", "weights"}

    def _rw_merge(self):
        for split in ["train", "val", "test"]:
            args = [
                "--config",
                str(Path(this_dir / "fixtures/test_config_rw.yaml")),
                "--rwm",
                "--split",
                split,
                *self.no,
            ]
            main(args)
            outpath = Path("tmp/upp-tests/integration/temp_workspace/test_out")
            assert (outpath / split).exists(), f"Output directory for split {split} not found"
            outfile = outpath / f"pp_output_{split}_vds.h5"
            assert outfile.exists()
            with h5py.File(outfile, "r") as f:
                assert "jets" in f, "Expected 'jets' group in output file"
                print("LOL", f.attrs, f["jets"].attrs, f["jets"].attrs.keys())

                assert (
                    "flavour_label" in f["jets"].attrs
                ), "Expected 'flavour_label' attribute in 'jets' group of output file"
                assert "flavour_label" in f["jets"].dtype.names

    def test_rw(self):
        self._run_split()
        self._calculate_weights()
        self._rw_merge()
