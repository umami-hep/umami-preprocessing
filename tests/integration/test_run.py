import os
from pathlib import Path
from types import SimpleNamespace

from ftag import get_mock_file
from ftag.hdf5 import H5Reader, H5Writer

from upp.main import run_pp


class TestClass:
    def generate_mock(self, out_file, N=100000):
        fname, f = get_mock_file(num_jets=2 * N)
        reader = H5Reader(fname, batch_size=10000, shuffle=False)
        variables = {"jets": None, "tracks": None}  # "None" means "all variables"
        out_fname = out_file
        writer = H5Writer(
            dst=out_fname,
            dtypes=reader.dtypes(variables),
            shapes=reader.shapes(N, groups=["jets", "tracks"]),
            shuffle=True,
        )
        for batch in reader.stream(variables=variables, num_jets=N):
            writer.write(batch)
        writer.close()

    def setup_method(self, method):
        os.makedirs("tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.generate_mock("tests/integration/temp_workspace/ntuples/data1.h5")
        self.generate_mock("tests/integration/temp_workspace/ntuples/data2.h5")
        print("setup_method      method:%s" % method.__name__)

    def teardown_method(self, method):
        pass
        # subprocess.run(
        #     ["rm", "-rf", "tests/integration/temp_workspace"],
        #     check=True,
        # )
        # print("teardown_method   method:%s" % method.__name__)

    def test_run(self):
        args = SimpleNamespace(
            config=Path("tests/integration/fixtures/test_conifig.yaml"),
            prep=True,
            resample=True,
            merge=True,
            norm=True,
            plot=True,
            split="train",
        )
        run_pp(args)
