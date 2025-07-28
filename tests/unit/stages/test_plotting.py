from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ftag import get_mock_file
from ftag.hdf5 import H5Reader

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.stages.plot import make_hist


class TestClass:
    def generate_mock(self, out_file, N=100):
        fname, f = get_mock_file(num_jets=N, fname=out_file)
        f.close()

    def setup_method(self, method):
        os.makedirs("tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.fname1 = "tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5"
        self.fname2 = "tmp/upp-tests/integration/temp_workspace/ntuples/data2.h5"
        self.generate_mock(out_file=self.fname1)
        self.generate_mock(out_file=self.fname2)

        # Get config
        self.config = PreprocessingConfig.from_file(
            Path(__file__).parent.parent.parent.resolve()
            / Path("integration/fixtures/test_config_pdf_auto.yaml"),
            "train",
        )

        # Get values dict
        self.values_dict = {
            "test": H5Reader(
                fname=self.fname1,
                batch_size=self.config.batch_size,
                jets_name=self.config.jets_name,
                shuffle=False,
                equal_jets=True,
            ).load(
                {
                    self.config.jets_name: [
                        "pt",
                        "abs_eta",
                        "mass",
                        "HadronConeExclTruthLabelID",
                    ]
                }
            )[self.config.jets_name]
        }
        print(f"setup_method, method: {method.__name__}")

    def teardown_method(self, method):
        subprocess.run(["rm", "-rf", "tmp/upp-tests/ubit"], check=True)
        print(f"teardown_method, method: {method.__name__}")

    def test_make_hist_initial(self):
        """Run the make_hist for the inital pT distribution."""
        make_hist(
            stage="initial",
            values_dict=self.values_dict,
            flavours=self.config.components.flavours,
            variable=self.config.sampl_cfg.vars[0],
            out_dir=Path("tmp/upp-tests/integration/temp_workspace/plots/"),
        )

    def test_make_hist_initial_no_pt(self):
        """Run the make_hist for the inital mass distribution."""
        make_hist(
            stage="initial",
            values_dict=self.values_dict,
            flavours=self.config.components.flavours,
            variable="mass",
            out_dir=Path("tmp/upp-tests/integration/temp_workspace/plots/"),
        )
