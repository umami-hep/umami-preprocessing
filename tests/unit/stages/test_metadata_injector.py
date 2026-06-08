from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import upp.stages.metadata_injector as mi


def _make_injector_file(tmp_path, jets_dtype, set_attrs=False):
    """Create a minimal HDF5 file for MetadataInjector tests."""
    fpath = tmp_path / "in.h5"
    n = 10
    jets = np.zeros(n, dtype=jets_dtype)
    if "mcEventWeight" in jets.dtype.names:
        jets["mcEventWeight"] = np.arange(n, dtype="f4") + 1.0
    sow = float(np.sum(jets["mcEventWeight"])) if "mcEventWeight" in jets.dtype.names else 55.0
    with h5py.File(fpath, "w") as f:
        ds = f.create_dataset("jets", data=jets)
        if set_attrs:
            ds.attrs["description"] = "test_attr"
            ds.attrs["version"] = 42
        cb = f.create_group("cutBookkeeper").create_group("nominal")
        cb.create_dataset("counts", data=np.array([sow], dtype="f8"))
        md = f.create_group("metadata").create_group("dummy_dsid")
        md.create_dataset("cross_section_pb", data=np.array(2.0))
        md.create_dataset("genFiltEff", data=np.array(0.5))
        md.create_dataset("kfactor", data=np.array(1.0))
    return fpath


class _Cfg:
    def __init__(self, path: Path) -> None:
        self.config = {"inputs": {"train": {"input_files": [str(path)]}}}


def _stub_finder(monkeypatch):
    class _Finder:
        def __init__(self, *_a, **_k) -> None:
            pass

        def inject_metadata(self) -> None:
            return None

    monkeypatch.setattr(mi, "MetadataFinder", _Finder, raising=False)


def test_metadata_injector_appends_physical_weight(tmp_path, monkeypatch):
    """Smoke-test MetadataInjector: physicalWeight is appended and fields are preserved."""
    _stub_finder(monkeypatch)
    fpath = _make_injector_file(tmp_path, [("mcEventWeight", "f4")])
    injector = mi.MetadataInjector(_Cfg(fpath))
    injector.run()

    with h5py.File(fpath) as f:
        out = f["jets"][:]
    assert "physicalWeight" in out.dtype.names
    assert "mcEventWeight" in out.dtype.names


def test_metadata_injector_missing_mcEventWeight_recovers(tmp_path, monkeypatch):
    """Exception path: missing mcEventWeight raises KeyError, backup is restored."""
    _stub_finder(monkeypatch)
    # jets without mcEventWeight triggers line 65 then lines 93-97
    fpath = _make_injector_file(tmp_path, [("someField", "f4")])
    backup_path = fpath.with_suffix(fpath.suffix + ".bak")

    injector = mi.MetadataInjector(_Cfg(fpath))
    injector.run()  # exception caught internally, must not propagate

    assert fpath.exists()
    assert not backup_path.exists()


def test_metadata_injector_drops_existing_physicalWeight(tmp_path, monkeypatch):
    """Existing physicalWeight field is dropped before recomputing (line 73)."""
    _stub_finder(monkeypatch)
    dtype = [("mcEventWeight", "f4"), ("physicalWeight", "f4")]
    fpath = _make_injector_file(tmp_path, dtype)
    # Overwrite physicalWeight with sentinel value to verify it gets replaced
    with h5py.File(fpath, "a") as f:
        data = f["jets"][:]
        data["physicalWeight"] = 999.0
        del f["jets"]
        f.create_dataset("jets", data=data)

    injector = mi.MetadataInjector(_Cfg(fpath))
    injector.run()

    with h5py.File(fpath) as f:
        out = f["jets"][:]
    assert "physicalWeight" in out.dtype.names
    assert not np.all(out["physicalWeight"] == 999.0)


def test_metadata_injector_preserves_dataset_attrs(tmp_path, monkeypatch):
    """Original dataset attrs are restored after jets dataset recreation (lines 84-85)."""
    _stub_finder(monkeypatch)
    fpath = _make_injector_file(tmp_path, [("mcEventWeight", "f4")], set_attrs=True)

    injector = mi.MetadataInjector(_Cfg(fpath))
    injector.run()

    with h5py.File(fpath) as f:
        assert f["jets"].attrs["description"] == "test_attr"
        assert f["jets"].attrs["version"] == 42
