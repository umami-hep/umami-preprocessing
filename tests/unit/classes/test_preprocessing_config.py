from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotmap import DotMap
from ftag import get_mock_file

from upp import __version__
from upp.classes.preprocessing_config import PreprocessingConfig


class TestPreprocessingConfig:
    def generate_mock(self, out_file, N=10):
        _, f = get_mock_file(num_jets=N, fname=out_file)
        f.close()

    def setup_method(self, method):
        os.makedirs("/tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.generate_mock("/tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5")
        self.generate_mock("/tmp/upp-tests/integration/temp_workspace/ntuples/data2.h5")
        print("setup_method      method:%s" % method.__name__)

    def teardown_method(self, method):
        subprocess.run(["rm", "-rf", "/tmp/upp-tests/integration"], check=True)
        print("teardown_method   method:%s" % method.__name__)

    @staticmethod
    def test_get_umami_general():
        config = PreprocessingConfig.from_file(
            Path("tests/unit/fixtures/test_conifig_pdf_auto_umami.yaml"),
            "train",
        )
        general = config.get_umami_general()
        assert general["dict_file"] == "dict/file/path.json"

    @staticmethod
    def test_get_umami_general_no_git():
        os.rename(".git", ".git_temp")
        config = PreprocessingConfig.from_file(
            Path("tests/unit/fixtures/test_conifig_pdf_auto_umami.yaml"),
            "train",
        )
        _ = config.get_umami_general()
        os.rename(".git_temp", ".git")
        assert config.git_hash == __version__

    @staticmethod
    def test_mimic_umami_config():
        config = PreprocessingConfig.from_file(
            Path("tests/unit/fixtures/test_conifig_pdf_auto_umami.yaml"),
            "train",
        )
        general = config.get_umami_general()
        general = DotMap(general, _dynamic=False)
        config.mimic_umami_config(general)

        assert config.general.dict_file == "dict/file/path.json"

    @staticmethod
    def test_mimic_umami_config_required():
        config = PreprocessingConfig.from_file(
            Path("tests/unit/fixtures/test_conifig_pdf_auto_umami_required.yaml"),
            "train",
        )
        general = config.get_umami_general()
        general = DotMap(general, _dynamic=False)
        config.mimic_umami_config(general)

        assert config.general.dict_file == "dict/file/path.json"

    @staticmethod
    def test_get_file_name():
        config = PreprocessingConfig.from_file(
            Path("tests/unit/fixtures/test_conifig_pdf_auto_umami.yaml"),
            "train",
        )

        assert (
            str(config.get_file_name("resampled"))
            == "/tmp/upp-tests/integration/temp_workspace/test_out/pp_output_train.h5"
        )
        assert str(config.get_file_name("resampled_scaled_shuffled")) == (
            "/tmp/upp-tests/integration/temp_workspace/test_out"
            + "/pp_output_train_resampled_scaled_shuffled.h5"
        )
