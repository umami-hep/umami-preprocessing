from __future__ import annotations

import pytest

from upp.classes.plotting_config import PlottingConfig


def test_plotting_config_labels():
    config = PlottingConfig(
        variable_labels={"pt_btagJes": "Custom pT"},
        sample_labels={"ttbar": "Top pair"},
    )

    assert config.variable_label("pt_btagJes") == "Custom pT"
    assert config.variable_label("unknown") == "unknown"
    assert config.sample_label("ttbar") == "Top pair"
    assert config.sample_label("unknown") == "unknown"


def test_plotting_config_default_pt_label():
    assert PlottingConfig().variable_label("pt_btagJes") == "$p_\\mathrm{T}$ [GeV]"


def test_plotting_config_default_mass_label():
    assert PlottingConfig().variable_label("mass") == "Jet Mass [GeV]"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_jets_plotting": 0}, "plotting.num_jets_plotting"),
        ({"output_formats": []}, "plotting.output_formats"),
        ({"linestyles": []}, "plotting.linestyles"),
    ],
)
def test_plotting_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PlottingConfig(**kwargs)
