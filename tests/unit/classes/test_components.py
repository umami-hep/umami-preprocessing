from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ftag import Cuts, Flavours, Sample
from ftag.mock import get_mock_file

import upp.classes.components as components_module
from upp.classes.components import Component, Components
from upp.classes.region import Region


def make_component(tmp_path: Path, vds_dir: Path | None) -> Component:
    fname = get_mock_file()[0]
    sample = Sample(pattern=fname, name="test", vds_dir=vds_dir)
    return Component(
        region=Region(name="lowpt", cuts=Cuts.empty()),
        sample=sample,
        flavour=Flavours.bjets,
        global_cuts=Cuts.empty(),
        dirname=tmp_path / "components" / "sub",
        num_global_objects=100,
        num_global_objects_estimate_available=0,
        equal_global_objects=True,
    )


def test_setup_reader_passes_vds_dir(tmp_path, monkeypatch):
    """setup_reader forwards vds_dir from sample to H5Reader when set."""
    vds_dir = tmp_path / "vds"
    comp = make_component(tmp_path, vds_dir=vds_dir)

    captured = {}

    class _H5Reader:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(components_module, "H5Reader", _H5Reader)
    comp.setup_reader(batch_size=100)

    assert "vds_dir" in captured
    assert captured["vds_dir"] == vds_dir


def test_setup_reader_no_vds_dir(tmp_path, monkeypatch):
    """setup_reader does not inject vds_dir when sample.vds_dir is None."""
    comp = make_component(tmp_path, vds_dir=None)

    captured = {}

    class _H5Reader:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(components_module, "H5Reader", _H5Reader)
    comp.setup_reader(batch_size=100)

    assert "vds_dir" not in captured


def _config_with_counts(tmp_path, counts):
    return SimpleNamespace(
        split="train",
        is_test=False,
        skip_checks=True,
        skip_auto_counts=False,
        skip_resampling=True,
        vds_dir=None,
        ntuple_dir=tmp_path,
        components_dir=tmp_path / "components" / "train",
        global_cuts=Cuts.empty(),
        flavour_cont=Flavours,
        num_global_objects_estimate_available=0,
        config={
            "components": [
                {
                    "region": {"name": "lowpt", "cuts": []},
                    "sample": {"name": "ttbar", "pattern": "data1.h5"},
                    "classes": ["bjets"],
                    "num_global_objects": count,
                }
                for count in counts
            ]
        },
    )


def test_from_config_mixed_auto_counts(tmp_path):
    config = _config_with_counts(tmp_path, ["auto", 100])
    with pytest.raises(ValueError, match="Only some components"):
        Components.from_config(config)


def test_from_config_skip_auto_counts(tmp_path):
    config = _config_with_counts(tmp_path, ["auto", "auto"])
    config.skip_auto_counts = True
    components = Components.from_config(config)

    assert all(c.num_global_objects == 0 for c in components)
