
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
from ftag.mock import get_mock_file, JET_VARS

JET_VARS += [('eventNumber', 'i4')]

from upp.main import main

this_dir = Path(__file__).parent
class TestRunRW:

    def generate_mock(self, out_file, N=1_000):
        _, f = get_mock_file(num_jets=N, fname=out_file)
        f['jets']['eventNumber'] = np.arange(N, dtype='i4')
        f.close()

    def setup_method(self, method):
        os.makedirs("tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5")
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data2.h5")
        self.generate_mock("tmp/upp-tests/integration/temp_workspace/ntuples/data3.h5")
        print("setup_method      method:%s" % method.__name__)

    def teardown_method(self, method):
        subprocess.run(["rm", "-r", "tmp"], check=True)
        print("teardown_method   method:%s" % method.__name__)

    def test_run_split(self):

        args = [
            "--config",
            str(Path(this_dir / "fixtures/test_config_rw.yaml")),
            "--no-resample",
            "--no-merge",
            "--no-norm",
            "--no-plot",
            "--split",
            "train",
            "--split-components",

        ]
        main(args)
        outpath = Path("tmp/upp-tests/integration/temp_workspace/split-components")
        for container in ['data1.h5', 'data2.h5', 'data3.h5']:
            assert (outpath / container).exists()
            component = 'highpt_zprime' if container=='data3.h5' else 'lowpt_ttbar'
            # We expect S * F number of output files, where S is the number of splits and F is the number of flavours
            exp_files = [
                f"{container}_{split}_{component}_{flavour}.h5"
                for split in ['train', 'val', 'test']
                for flavour in ['bjets', 'cjets', 'ujets', 'taujets']
            ]
            for exp_file in exp_files:
                assert (outpath / container / exp_file).exists(), f"Expected file {exp_file} not found in {outpath}"
        
        