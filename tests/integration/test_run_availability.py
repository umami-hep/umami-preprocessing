from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from ftag import get_mock_file

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.main import main as preprocess
from upp.utils.availability import CACHE_FNAME, estimate_availability, main, recommend

this_dir = Path(__file__).parent
CONFIG = this_dir / "fixtures/test_config_availability.yaml"
AUTO_CONFIG = this_dir / "fixtures/test_config_auto_counts.yaml"
CACHE = Path("tmp/upp-tests/integration/temp_workspace/test_out_availability") / CACHE_FNAME


class TestClass:
    def setup_method(self, _method):
        os.makedirs("tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        for fname in ("data1.h5", "data2.h5"):
            _, f = get_mock_file(
                num_jets=100_000,
                fname=f"tmp/upp-tests/integration/temp_workspace/ntuples/{fname}",
            )
            f.close()

    def teardown_method(self, _method):
        subprocess.run(["rm", "-r", "tmp"], check=True)

    def test_estimate_matches_ftag(self):
        config = PreprocessingConfig.from_file(CONFIG, "train", skip_config_copy=True)
        component = config.components[0]
        component.setup_reader(config.batch_size, global_name=config.global_name)

        expected = component.reader.estimate_available_global_objects(component.cuts, None)
        estimate = estimate_availability(component, {"train": component.cuts}, None)
        assert estimate["train"] == expected

    def test_recommend(self):
        available, counts = recommend(CONFIG)

        assert set(counts) == {"train", "val", "test"}
        for split, split_counts in counts.items():
            # ttbar has a sample weight of two, so lowpt gets twice the objects of highpt
            for region, name in split_counts:
                if region == "lowpt":
                    assert split_counts[(region, name)] == 2 * split_counts[("highpt", name)]
                assert split_counts[(region, name)] <= available[split][(region, name)]

    def test_cache_is_reused(self, capsys):
        main(["--config", str(CONFIG)])
        assert CACHE.exists()
        cached = yaml.safe_load(CACHE.read_text())
        assert set(cached["components"]) == {
            c.name
            for c in PreprocessingConfig.from_file(
                CONFIG, "train", skip_config_copy=True
            ).components
        }

        main(["--config", str(CONFIG)])
        assert "Using cached estimate" in capsys.readouterr().out

    def test_auto_counts(self):
        main(["--config", str(AUTO_CONFIG)])
        _, counts = recommend(AUTO_CONFIG)

        for split in ("train", "val", "test"):
            config = PreprocessingConfig.from_file(AUTO_CONFIG, split, skip_config_copy=True)
            for component in config.components:
                key = (component.region.name, component.flavour.name)
                assert component.num_global_objects == counts[split][key]
                assert component.num_global_objects > 0

    def test_auto_counts_without_estimate(self):
        with pytest.raises(ValueError, match="estimate_object_counts"):
            PreprocessingConfig.from_file(AUTO_CONFIG, "train", skip_config_copy=True)

    def test_auto_counts_preprocessing(self):
        main(["--config", str(AUTO_CONFIG)])
        preprocess(
            [
                "--config",
                str(AUTO_CONFIG),
                "--no-plot",
                "--split",
                "train",
            ]
        )
