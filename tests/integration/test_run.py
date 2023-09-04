import subprocess
from pathlib import Path
from types import SimpleNamespace

from upp.main import run_pp


class TestClass:
    def generate_mock(self):
        pass

    def setup_method(self, method):
        print("setup_method      method:%s" % method.__name__)

    def teardown_method(self, method):
        print("teardown_method   method:%s" % method.__name__)

    def test_1(self):
        # subprocess.run(
        #     [
        #         "python",
        #         "upp/main.py",
        #         "--config",
        #         "tests/integration/fixtures/test_conifig.yaml",
        #     ],
        #     capture_output=True,
        #     text=True,
        #     check=True,
        # )
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
