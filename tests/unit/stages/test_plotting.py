from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ftag import get_mock_file

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.stages.plot import make_hist


class TestClass:
    def generate_mock(self, out_file, N=100):
        fname, f = get_mock_file(num_jets=N, fname=out_file)
        f.close()

    def setup_method(self, method):
        os.makedirs("tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5")
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data2.h5")
        print("setup_method      method:%s" % method.__name__)

    def teardown_method(self, method):
        subprocess.run(["rm", "-rf", "tmp/upp-tests/ubit"], check=True)
        print("teardown_method   method:%s" % method.__name__)

    def test_make_hist_initial(self):
        config = PreprocessingConfig.from_file(
            Path(__file__).parent.parent.parent.resolve()
            / Path("integration/fixtures/test_config_pdf_auto.yaml"),
            "train",
        )
        make_hist(
            stage="initial",
            flavours=config.components.flavours,
            variable=config.sampl_cfg.vars[0],
            in_paths_list=["tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5"],
        )

    def test_make_hist_initial_no_pt(self):
        config = PreprocessingConfig.from_file(
            Path(__file__).parent.parent.parent.resolve()
            / Path("integration/fixtures/test_config_pdf_auto.yaml"),
            "train",
        )
        make_hist(
            stage="initial",
            flavours=config.components.flavours,
            variable="mass",
            in_paths_list=["tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5"],
        )
