from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from ftag import Cuts, Flavours, get_mock_file
from ftag.hdf5 import H5Reader

import upp.stages.plot as plot_mod
from upp.classes.preprocessing_config import PreprocessingConfig
from upp.stages.plot import make_hist


class TestClass:
    def generate_mock(self, out_file, N=100):
        _fname, f = get_mock_file(num_jets=N, fname=out_file)
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
        out_dir = Path("tmp/upp-tests/integration/temp_workspace/plots/")
        make_hist(
            stage="initial",
            values_dict=self.values_dict,
            flavours=self.config.components.flavours,
            variable=self.config.sampl_cfg.vars[0],
            out_dir=out_dir,
        )

        assert (out_dir / f"initial_{self.config.sampl_cfg.vars[0]}.pdf").exists()
        assert (out_dir / f"initial_{self.config.sampl_cfg.vars[0]}.png").exists()

    def test_make_hist_initial_no_pt(self):
        """Run the make_hist for the inital mass distribution."""
        make_hist(
            stage="initial",
            values_dict=self.values_dict,
            flavours=self.config.components.flavours,
            variable="mass",
            out_dir=Path("tmp/upp-tests/integration/temp_workspace/plots/"),
        )


def test_plot_helpers_format_labels_and_ranges():
    """Check compact labels and pT unit conversion helpers."""
    assert plot_mod._format_num_jets(999) == "999"
    assert plot_mod._format_num_jets(100_000) == "100k"
    assert plot_mod._format_num_jets(10_000_000) == "10M"
    assert (
        plot_mod._atlas_second_tag("ttbar", "zprime", num_jets=100_000)
        == "$\\sqrt{s} = 13/13.6$ TeV, $t\\bar{t}$ + $Z'$ jets\n100k jets"
    )
    assert plot_mod._display_range("pt_btagJes", (20_000, 250_000)) == (20, 250)
    assert plot_mod._display_range("absEta_btagJes", (0, 2.5)) == (0, 2.5)


def test_plot_helpers_pt_regions_and_stitching():
    """Check that region and stitching windows are derived from pT cuts."""
    low = plot_mod.PlotRegion(
        "lowpt",
        Cuts.from_list([["pt_btagJes", ">", 20_000], ["pt_btagJes", "<", 250_000]]),
        (20_000, 250_000),
    )
    high = plot_mod.PlotRegion(
        "highpt",
        Cuts.from_list([["pt_btagJes", ">", 250_000], ["pt_btagJes", "<", 6_000_000]]),
        (250_000, 6_000_000),
    )

    full = plot_mod._full_region([low, high], "pt_btagJes")
    assert full.name == "full"
    assert full.pt_range == (20_000, 6_000_000)

    stitching = plot_mod._stitching_regions([high, low], "pt_btagJes")
    assert len(stitching) == 1
    assert stitching[0].name == "stitching"
    assert stitching[0].pt_range == (150_000, 350_000)


def test_post_resampling_paths_split_mode(tmp_path):
    """Check split-output post-resampling globs, including per-sample test output."""
    sample = SimpleNamespace(name="ttbar")
    components = SimpleNamespace(samples=[sample])
    config = SimpleNamespace(
        out_fname=tmp_path / "pp_output_test.h5",
        split="test",
        merge_test_samples=False,
        num_jets_per_output_file=10,
        components=components,
    )

    paths = plot_mod._post_resampling_paths(config, "test")
    assert paths == [tmp_path / "test" / "pp_output_test_ttbar*.h5"]

    config.merge_test_samples = True
    paths = plot_mod._post_resampling_paths(config, "test")
    assert paths == [tmp_path / "test" / "pp_output_test*.h5"]


def test_plot_initial_uses_split_suffix_and_plotting_jet_count(monkeypatch, tmp_path):
    """Check initial plot calls include split-specific suffixes and plotting counts."""

    class FakeComponents:
        def __init__(self):
            self.flavours = [Flavours["bjets"]]

        def groupby_sample(self):
            sample = SimpleNamespace(name="ttbar", path=[tmp_path / "dummy.h5"])
            return [(sample, FakeSampleComponents())]

    class FakeSampleComponents:
        def groupby_region(self):
            region = SimpleNamespace(
                name="lowpt",
                cuts=Cuts.from_list([["pt", ">", 20_000], ["pt", "<", 250_000]]),
            )
            region_components = SimpleNamespace(
                flavours=[Flavours["bjets"]],
                num_jets=100_000,
            )
            return [(region, region_components)]

    config = SimpleNamespace(
        split="val",
        sampl_cfg=SimpleNamespace(
            vars=["pt"],
            bins={"pt": [[20_000, 250_000, 5]]},
        ),
        components=FakeComponents(),
        num_jets_estimate_plotting=10_000,
        jets_name="jets",
        batch_size=100,
        out_dir=tmp_path,
    )

    calls = []

    def fake_load_jets(_config, _in_paths, _vars_to_load):
        return object()

    def fake_make_hist(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(plot_mod, "_load_jets", fake_load_jets)
    monkeypatch.setattr(plot_mod, "make_hist", fake_make_hist)

    plot_mod._plot_initial(config)

    assert len(calls) == 1
    assert calls[0]["suffix"] == "_val_ttbar_lowpt"
    assert calls[0]["bins_range"] == (20, 250)
    assert calls[0]["atlas_second_tag"] == ("$\\sqrt{s} = 13/13.6$ TeV, $t\\bar{t}$ jets\n10k jets")
