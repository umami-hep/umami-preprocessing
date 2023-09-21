from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from ftag import get_mock_file

from upp.main import run_pp

this_dir = Path(__file__).parent


class TestClass:
    def generate_mock(self, out_file, N=100_000):
        fname, f = get_mock_file(num_jets=N, fname=out_file)
        f.close()

    def setup_method(self, method):
        os.makedirs("/tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.generate_mock("/tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5")
        self.generate_mock("/tmp/upp-tests/integration/temp_workspace/ntuples/data2.h5")
        print("setup_method      method:%s" % method.__name__)

    def teardown_method(self, method):
        subprocess.run(["rm", "-rf", "/tmp/upp-tests/integration"], check=True)
        print("teardown_method   method:%s" % method.__name__)

    def test_run_pdf_auto(self):
        args = SimpleNamespace(
            config=Path(this_dir / "fixtures/test_conifig_pdf_auto.yaml"),
            prep=True,
            resample=True,
            merge=True,
            norm=True,
            plot=True,
            split="train",
        )
        run_pp(args)

    def test_run_pdf_upscale(self):
        args = SimpleNamespace(
            config=Path(this_dir / "fixtures/test_conifig_pdf_upscaled.yaml"),
            prep=True,
            resample=True,
            merge=False,
            norm=False,
            plot=False,
            split="train",
        )
        run_pp(args)

    def test_run_countup(self):
        args = SimpleNamespace(
            config=Path(this_dir / "fixtures/test_conifig_countup.yaml"),
            prep=True,
            resample=True,
            merge=False,
            norm=False,
            plot=False,
            split="train",
        )
        run_pp(args)
