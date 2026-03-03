from __future__ import annotations

from pathlib import Path

# import pytest
from ftag import Cuts, Flavours, Sample
from ftag.mock import get_mock_file

import upp.classes.components as components_module
from upp.classes.components import Component
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
        num_jets=100,
        num_jets_estimate_available=0,
        equal_jets=True,
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
