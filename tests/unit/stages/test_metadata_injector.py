from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import upp.stages.metadata_injector as mi


def test_metadata_injector_appends_physical_weight(tmp_path, monkeypatch):
    """Smoke-test MetadataInjector when dependencies are available.

    In CI environments where `MetadataFinder` is not available via the installed `ftag`,
    importing this module should fail; in that case we skip.
    """

    # Stub MetadataFinder to avoid depending on external metadata DB.
    class _Finder:
        def __init__(self, *_a, **_k):
            pass

        def inject_metadata(self):
            return None

    monkeypatch.setattr(mi, "MetadataFinder", _Finder, raising=False)

    # Create minimal input file.
    fpath = tmp_path / "in.h5"
    n = 10
    jets = np.zeros(n, dtype=[("mcEventWeight", "f4")])
    jets["mcEventWeight"] = np.arange(n, dtype="f4") + 1.0
    with h5py.File(fpath, "w") as f:
        f.create_dataset("jets", data=jets)
        cb = f.create_group("cutBookkeeper").create_group("nominal")
        # MetadataInjector supports both a structured and a plain dataset here. Use a
        # plain float dataset to keep the test robust across h5py versions.
        cb.create_dataset(
            "counts", data=np.array([float(np.sum(jets["mcEventWeight"]))], dtype="f8")
        )
        md = f.create_group("metadata").create_group("dummy_dsid")
        md.create_dataset("cross_section_pb", data=np.array(2.0))
        md.create_dataset("genFiltEff", data=np.array(0.5))
        md.create_dataset("kfactor", data=np.array(1.0))

    class _Cfg:
        def __init__(self, path: Path):
            self.config = {"inputs": {"train": {"input_files": [str(path)]}}}

    injector = mi.MetadataInjector(_Cfg(fpath))
    injector.run()

    with h5py.File(fpath) as f:
        out = f["jets"][:]
    assert "physicalWeight" in out.dtype.names
    # Also ensure old fields remain.
    assert "mcEventWeight" in out.dtype.names
