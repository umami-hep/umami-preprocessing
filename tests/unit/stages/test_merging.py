"""
Unit-tests (no disk IO) for upp.stages.merging.Merging.

We monkey-patch upp.stages.merging.H5Writer with an in-memory stub so that
_open_writer / write_chunk / write_components can be exercised without
creating real HDF5 files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import h5py
import numpy as np
from ftag import Flavours

import upp.stages.merging as merging_mod
from upp.classes.preprocessing_config import PreprocessingConfig


# Stub H5Writer that just records calls, does NO real IO
class MemWriter:
    def __init__(self, dst, dtypes, shapes, jets_name, **_):
        self.dst = Path(dst)
        self.dtypes = dtypes
        self.shapes = shapes
        self.num_jets = next(iter(shapes.values()))[0]
        self.jets_name = jets_name
        self.num_written = 0
        self.attrs = {}

    # API that Merging calls
    def write(self, data: dict):
        self.num_written += len(data[self.jets_name])

    def close(self):
        assert self.num_written == self.num_jets

    def add_attr(self, name, value, *_):
        self.attrs[name] = value


# Minimal helpers to fabricate Components / Variables for unit tests
def _jets_struct(n):
    """Return a (n,) structured array with two float vars."""
    dtype = np.dtype([("pt", "f4"), ("eta", "f4")])
    rng = np.random.default_rng(42)
    arr = np.empty(n, dtype=dtype)
    arr["pt"] = rng.random(n)
    arr["eta"] = rng.random(n)
    return arr


class DummyComponent(SimpleNamespace):
    """Dummy component for testing.

    Simulates the subset of the real Component interface that Merging uses:
    - flavour
    - stream  (generator of batches)
    - num_jets
    - complete flag
    """

    def __init__(self, flavour, jet_batches):
        super().__init__()
        self.flavour = Flavours[flavour]
        self._batches = list(jet_batches)
        self.num_jets = sum(len(b["jets"]) for b in self._batches)
        self.complete = False
        self.out_path = Path("/dev/null")

    def setup_reader(self, *_args, **_kw):
        pass

    @property
    def stream(self):
        def _gen():
            yield from self._batches

        return _gen()


def _minimal_merging(monkeypatch, jets_per_file=10) -> merging_mod.Merging:
    """Create modded Merging version.

    Return a fully-initialised Merging instance whose writer is the MemWriter
    and whose config / variables are minimal SimpleNamespace objects.
    """
    # Patch H5Writer
    monkeypatch.setattr(merging_mod, "H5Writer", MemWriter)

    # Minimal Variables object
    variables = SimpleNamespace(
        variables=["jets"],
        selectors={},
        combined=lambda: {"jets": None},
        keys=lambda: ["jets"],
    )

    # Setup cfg & components
    cfg = SimpleNamespace(
        components=SimpleNamespace(flavours=[Flavours["bjets"]]),
        variables=variables,
        batch_size=100,
        jets_name="jets",
        num_jets_per_output_file=jets_per_file,
        file_tag="split",
        out_fname=Path("/tmp/merged.h5"),
        git_hash="deadbeef",
        config={},
        is_test=False,
        merge_test_samples=False,
    )

    return merging_mod.Merging(cast(PreprocessingConfig, cfg))


# Actual tests
def test_add_jet_flavour_label(monkeypatch):
    """A flavour label is added exactly once and matches the flavour list."""
    merge = _minimal_merging(monkeypatch, jets_per_file=7)

    jets = _jets_struct(5)
    comp = SimpleNamespace(flavour=Flavours["bjets"])
    tagged = merge.add_jet_flavour_label(jets, comp)

    assert "flavour_label" in tagged.dtype.names
    assert np.all(tagged["flavour_label"] == 0)

    # Calling again must not duplicate the column
    same = merge.add_jet_flavour_label(tagged, comp)
    assert same.dtype == tagged.dtype


def test_open_writer_names_and_shapes(monkeypatch):
    """_open_writer should create a MemWriter with the correct dst name and shapes."""
    merge = _minimal_merging(monkeypatch, jets_per_file=7)

    # Minimal shapes / dtypes upfront
    merge.base_shapes = {"jets": (42,)}
    merge.dtypes = {"jets": _jets_struct(1).dtype}

    merge._open_writer(
        sample=None,
        jets_in_file=7,
        file_idx=0,
        components=SimpleNamespace(unique_jets=True, jet_counts={}, dsids=[]),
    )

    writer = merge.writer
    assert isinstance(writer, MemWriter)
    assert writer.num_jets == 7
    assert writer.dst.name.startswith("merged_split_000")
    assert "flavour_label" in writer.attrs


def test_write_chunk_splits(monkeypatch):
    """Test split writing of write_chunk.

    With num_jets_per_output_file=5 and a batch of 8 jets, write_chunk must
    create two MemWriter instances and write all jets.
    """
    merge = _minimal_merging(monkeypatch, jets_per_file=5)

    # Prepare internal bookkeeping identical to write_components()
    jets1 = {"jets": _jets_struct(8)}
    comp = DummyComponent("bjets", [jets1])
    comps = [comp]

    # Fake dtypes / shapes and open first writer
    merge.dtypes = {"jets": jets1["jets"].dtype}
    merge.base_shapes = {"jets": (8,)}
    merge.total_jets = 8
    merge._file_idx = 0
    merge.jets_written = 0
    merge.current_components = SimpleNamespace(unique_jets=True, jet_counts={}, dsids=[])
    merge._sample = None

    merge._open_writer(None, 5, 0, merge.current_components)

    # Run write_chunk once
    n_written = merge.write_chunk(comps)

    # Assertions
    assert n_written == 8

    # First writer closed & second opened
    assert merge._file_idx == 1
    assert merge.writer.num_written == 3


def test_write_chunk_rollover(monkeypatch):
    """Check the rollover behaviour.

    When the current MemWriter is already full (capacity_left == 0)
    `write_chunk` must

    1. close that writer,
    2. open a fresh one (file_idx increments),
    3. write the incoming batch into the new file,
    4. update jets_written.
    """
    # Create a merger with 5 jets / file
    merge = _minimal_merging(monkeypatch, jets_per_file=5)

    # Incoming batch: 3 jets
    batch = {"jets": _jets_struct(3)}
    comp = DummyComponent("bjets", [batch])
    comps = [comp]

    # Fake the bookkeeping exactly as write_components() would
    merge.dtypes = {"jets": batch["jets"].dtype}
    merge.base_shapes = {"jets": (8,)}
    merge.total_jets = 8
    merge.jets_written = 5
    merge._file_idx = 0
    merge._sample = None
    merge.current_components = SimpleNamespace(unique_jets=True, jet_counts={}, dsids=[])

    # Open the first writer with capacity 5 and mark it as "full"
    merge._open_writer(None, 5, 0, merge.current_components)
    merge.writer.num_written = 5

    # Call write_chunk
    n = merge.write_chunk(comps)

    assert n == 3
    assert merge._file_idx == 1
    assert merge.writer.num_written == 3
    assert merge.jets_written == 8


def test_write_chunk_returns_zero_when_no_space_left(monkeypatch):
    """Check early exit."""
    merge = _minimal_merging(monkeypatch, jets_per_file=4)

    # One dummy component that would supply more jets
    jets_batch = {"jets": _jets_struct(3)}
    comp = DummyComponent("bjets", [jets_batch])
    comps = [comp]

    # Mimic state after everything has already been written
    merge.total_jets = 4
    merge.jets_written = 4
    merge._file_idx = 0
    merge.current_components = SimpleNamespace(unique_jets=True, jet_counts={}, dsids=[])
    merge._sample = None

    # We still need valid dtypes / shapes for _open_writer
    merge.dtypes = {"jets": jets_batch["jets"].dtype}
    merge.base_shapes = {"jets": (4,)}

    # Open a writer that is already full (num_written == num_jets)
    merge._open_writer(None, 4, 0, merge.current_components)
    merge.writer.num_written = merge.writer.num_jets

    n_written = merge.write_chunk(comps)

    # it must early-return with 0 and NOT raise or open new writers
    assert n_written == 0

    # _file_idx is incremented once, but _open_writer was not called again
    assert merge._file_idx == 1
    assert merge.writer.num_written == merge.writer.num_jets


class ReaderStub:
    """Minimal reader stub used by write_components()."""

    def __init__(self, batches: list[dict[str, np.ndarray]]):
        self._batches = batches
        self.num_jets = sum(len(b["jets"]) for b in batches)

    def dtypes(self, _vars):
        # Assume at least one batch exists
        return {"jets": self._batches[0]["jets"].dtype}

    def shapes(self, total_jets: int, _keys):
        # Base-shapes are used only for dataset names and leading dim
        return {"jets": (total_jets,)}

    def stream(self, _vars, _num_jets):
        def _gen():
            yield from self._batches

        return _gen()


class ComponentStub:
    """Component with a ReaderStub; compatible with write_components()."""

    def __init__(self, flavour: str, batches: list[dict[str, np.ndarray]]):
        from ftag import Flavours

        self.flavour = Flavours[flavour]
        self._batches = batches
        self.num_jets = sum(len(b["jets"]) for b in batches)
        self.complete = False
        self.out_path = Path("/dev/null")
        self.reader = None

    def setup_reader(self, *_args, **_kwargs):
        self.reader = ReaderStub(self._batches)


class ComponentsStub:
    """Container mimicking the 'Components' interface used by Merging."""

    def __init__(self, comps: list[ComponentStub]):
        self._comps = comps
        self.num_jets = sum(c.num_jets for c in comps)
        self.unique_jets = True
        self.jet_counts: dict[str, int] = {}
        self.dsids: list[int] = []

    def __iter__(self):
        return iter(self._comps)

    def __getitem__(self, i: int):
        return self._comps[i]


def _mk_merge_for_path(monkeypatch, out_path: Path, jets_per_file=5):
    """Like _minimal_merging, but lets us control out_fname path on disk."""
    # Patch H5Writer -> MemWriter for *output* (no real IO during merging)
    monkeypatch.setattr(merging_mod, "H5Writer", MemWriter)

    variables = SimpleNamespace(
        variables=["jets"],
        selectors={},  # keep selectors empty for simplicity
        combined=lambda: {"jets": None},
        keys=lambda: ["jets"],
    )
    cfg = SimpleNamespace(
        components=SimpleNamespace(flavours=[Flavours["bjets"]]),
        variables=variables,
        batch_size=100,
        jets_name="jets",
        num_jets_per_output_file=jets_per_file,
        file_tag="split",
        out_fname=out_path,
        git_hash="deadbeef",
        config={},
        is_test=False,
        merge_test_samples=False,
    )
    merge = merging_mod.Merging(cast(PreprocessingConfig, cfg))
    return merge


def _write_valid_part(fname: Path, rows: int):
    """Create a minimal valid HDF5 part with dataset 'jets' of length `rows`."""
    fname.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(fname, "w") as f:
        # Single 1D float dataset is enough for validator
        dset = f.create_dataset("jets", shape=(rows,), dtype="f4")
        dset[...] = 0.0


def _write_invalid_hdf5(fname: Path):
    """Write non-HDF5 bytes so h5py fails to open (simulating a corrupt file)."""
    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, "wb") as fd:
        fd.write(b"NOT AN HDF5 FILE")


# --- New tests for the new functionality ---


def test_part_fname_formatting(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=7)

    p0 = merge._part_fname(sample=None, file_idx=0)
    p7 = merge._part_fname(sample=None, file_idx=7)
    ps = merge._part_fname(sample="ttbar", file_idx=12)

    assert p0.name.endswith("merged_split_000.h5")
    assert p7.name.endswith("merged_split_007.h5")
    # name should still be the same, only directory changes with sample
    assert ps.name.endswith("merged_ttbar_split_012.h5")
    assert ps.parent == tmp_path


def test_detect_and_clean_completed_parts_empty_dir(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 10
    merge.base_shapes = {"jets": (10,)}
    # no files exist
    idx = merge._detect_and_clean_completed_parts(None)
    assert idx == 0


def test_expected_rows_for_part_middle_and_tail(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 12  # 5 + 5 + 2
    assert merge._expected_rows_for_part(0) == 5
    assert merge._expected_rows_for_part(1) == 5
    assert merge._expected_rows_for_part(2) == 2
    assert merge._expected_rows_for_part(3) == 0


def test_is_part_valid_happy_path(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 12
    merge.base_shapes = {"jets": (12,)}  # dataset names only; length unused here

    f0 = merge._part_fname(None, 0)
    _write_valid_part(f0, rows=5)

    assert merge._is_part_valid(None, 0) is True


def test_is_part_valid_multiple_datasets(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 5
    merge.base_shapes = {"jets": (5,), "tracks": (5,)}

    f0 = merge._part_fname(None, 0)
    with h5py.File(f0, "w") as f:
        f.create_dataset("jets", shape=(5,), dtype="f4")
        f.create_dataset("tracks", shape=(5,), dtype="f4")
    assert merge._is_part_valid(None, 0) is True


def test_is_part_valid_wrong_length(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 12
    merge.base_shapes = {"jets": (12,)}

    f1 = merge._part_fname(None, 1)
    _write_valid_part(f1, rows=3)  # expected 5
    assert merge._is_part_valid(None, 1) is False


def test_is_part_valid_missing_jets_dataset(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 5
    merge.base_shapes = {"jets": (5,)}

    f0 = merge._part_fname(None, 0)
    with h5py.File(f0, "w") as f:
        f.create_dataset("something_else", shape=(5,), dtype="f4")
    assert merge._is_part_valid(None, 0) is False


def test_is_part_valid_mismatched_lengths(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 5
    # Declare two expected datasets; only those present are compared
    merge.base_shapes = {"jets": (5,), "tracks": (5,)}

    f0 = merge._part_fname(None, 0)
    with h5py.File(f0, "w") as f:
        f.create_dataset("jets", shape=(5,), dtype="f4")
        f.create_dataset("tracks", shape=(4,), dtype="f4")  # mismatched
    assert merge._is_part_valid(None, 0) is False


def test_is_part_valid_corrupt_file(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 5
    merge.base_shapes = {"jets": (5,)}

    f0 = merge._part_fname(None, 0)
    _write_invalid_hdf5(f0)
    assert merge._is_part_valid(None, 0) is False


def test_detect_and_clean_completed_parts_counts_and_deletes(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 13  # parts: [5, 5, 3]
    merge.base_shapes = {"jets": (13,)}

    # Create two valid parts 0 and 1
    _write_valid_part(merge._part_fname(None, 0), rows=5)
    _write_valid_part(merge._part_fname(None, 1), rows=5)

    # Create a corrupt part 2
    bad = merge._part_fname(None, 2)
    _write_invalid_hdf5(bad)
    assert bad.exists()

    # auto_fix_parts=True: should delete 'bad' and return idx=2
    merge.auto_fix_parts = True
    idx = merge._detect_and_clean_completed_parts(None)
    assert idx == 2
    assert not bad.exists()

    # If we don't create any further part, the next run should stop at 2
    idx2 = merge._detect_and_clean_completed_parts(None)
    assert idx2 == 2  # still first missing


def test_detect_and_clean_completed_parts_no_delete_when_disabled(monkeypatch, tmp_path):
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 10
    merge.base_shapes = {"jets": (10,)}

    _write_valid_part(merge._part_fname(None, 0), rows=5)
    bad = merge._part_fname(None, 1)
    _write_invalid_hdf5(bad)

    merge.auto_fix_parts = False
    idx = merge._detect_and_clean_completed_parts(None)
    assert idx == 1
    assert bad.exists()  # left in place


def test_detect_and_clean_handles_unlink_error(monkeypatch, tmp_path):
    """Simulate OSError during unlink to cover error logging branch."""
    merge = _mk_merge_for_path(monkeypatch, tmp_path / "merged.h5", jets_per_file=5)
    merge.total_jets = 10
    merge.base_shapes = {"jets": (10,)}

    # Valid part 0
    _write_valid_part(merge._part_fname(None, 0), rows=5)
    # Corrupt part 1
    bad = merge._part_fname(None, 1)
    _write_invalid_hdf5(bad)

    # Make Path.unlink raise OSError for this file
    original_unlink = Path.unlink

    def _unlink_raise(self):
        if self == bad:
            raise OSError("permission denied")
        return original_unlink(self)

    monkeypatch.setattr(Path, "unlink", _unlink_raise)

    merge.auto_fix_parts = True
    idx = merge._detect_and_clean_completed_parts(None)
    # It stops at first invalid (failed to delete), index unchanged and file remains
    assert idx == 1
    assert bad.exists()


def test_resume_skips_completed_parts_and_opens_next(monkeypatch, tmp_path):
    """End-to-end resume: pre-create parts 0 & 1; merging should start at part 2 with capacity 3."""
    out = tmp_path / "merged.h5"
    merge = _mk_merge_for_path(monkeypatch, out, jets_per_file=5)

    # One component with 13 jets total, e.g. [5,5,3] split
    all_jets = {"jets": _jets_struct(13)}
    batches = [all_jets]
    comp = ComponentStub("bjets", batches)
    comps = ComponentsStub([comp])

    # Pre-create valid parts 0 and 1
    _write_valid_part(merge._part_fname(None, 0), rows=5)
    _write_valid_part(merge._part_fname(None, 1), rows=5)

    # Spy on _open_writer to ensure it's NOT called during fast-forward
    open_calls = []
    _orig_open = merging_mod.Merging._open_writer

    def _wrapped_open(self, sample, jets_in_file, file_idx, components):
        # Fast-forward must be finished when we open a real writer
        assert self._fast_forwarding is False
        open_calls.append((sample, jets_in_file, file_idx))
        return _orig_open(self, sample, jets_in_file, file_idx, components)

    monkeypatch.setattr(merging_mod.Merging, "_open_writer", _wrapped_open)

    # Run merge
    merge.write_components(sample=None, components=comps)

    # We expect the first (and only) open to be for part index 2 with capacity 3
    assert len(open_calls) >= 1
    _, jets_in_file, file_idx = open_calls[0]
    assert file_idx == 2
    assert jets_in_file == 3

    # The MemWriter should have written exactly 3 jets in that last file
    assert isinstance(merge.writer, MemWriter)
    assert merge.writer.num_jets == 3
    assert merge.writer.num_written == 3


def test_fast_forward_does_not_open_real_writer(monkeypatch, tmp_path):
    """Ensure no real MemWriter is opened during fast-forward."""
    out = tmp_path / "merged.h5"
    merge = _mk_merge_for_path(monkeypatch, out, jets_per_file=4)

    # Total jets 9 -> parts [4,4,1]; pre-create part 0 only
    total = 9
    jets = {"jets": _jets_struct(total)}
    comp = ComponentStub("bjets", [jets])
    comps = ComponentsStub([comp])

    _write_valid_part(merge._part_fname(None, 0), rows=4)

    # Count how many times _open_writer is called. It should be called:
    # - once after fast-forward (to open part 1),
    # - possibly again if rollover happens when writing.
    call_times = {"count": 0}
    _orig_open = merging_mod.Merging._open_writer

    def _wrapped_open(self, sample, jets_in_file, file_idx, components):
        # Must never be invoked while fast-forwarding
        assert self._fast_forwarding is False
        call_times["count"] += 1
        return _orig_open(self, sample, jets_in_file, file_idx, components)

    monkeypatch.setattr(merging_mod.Merging, "_open_writer", _wrapped_open)

    merge.write_components(None, comps)

    # At least one open (for part 1) happened, and none during fast-forward
    assert call_times["count"] >= 1


def test_nullwriter_basic_behaviour():
    """Cover the tiny _NullWriter helper."""
    nw = merging_mod.Merging._NullWriter(capacity=5)
    # add_attr and close are no-ops, but execute them for coverage
    nw.add_attr("x", 1)
    nw.close()
    # write smaller than capacity
    batch1 = {"jets": _jets_struct(3)}
    nw.write(batch1)
    assert nw.num_written == 3
    # write beyond capacity (should clamp)
    batch2 = {"jets": _jets_struct(4)}
    nw.write(batch2)
    assert nw.num_written == 5


def test_write_chunk_all_components_complete_early_return(monkeypatch):
    """If all components exhaust immediately, write_chunk returns 0."""
    merge = _minimal_merging(monkeypatch, jets_per_file=5)

    # No batches → StopIteration immediately
    comp = DummyComponent("bjets", [])
    comps = [comp]

    # Minimal bookkeeping to allow a writer
    jets0 = _jets_struct(0)
    merge.dtypes = {"jets": jets0.dtype}
    merge.base_shapes = {"jets": (0,)}
    merge.total_jets = 0
    merge._file_idx = 0
    merge.jets_written = 0
    merge.current_components = SimpleNamespace(unique_jets=True, jet_counts={}, dsids=[])
    merge._sample = None
    merge._open_writer(None, 0, 0, merge.current_components)

    n = merge.write_chunk(comps)
    assert n == 0


def test_write_components_single_file_mode(monkeypatch, tmp_path):
    """When num_jets_per_output_file is None, no split suffix is added."""
    out = tmp_path / "merged.h5"
    merge = _mk_merge_for_path(monkeypatch, out, jets_per_file=None)
    merge.num_jets_per_output_file = None  # ensure single-file mode

    jets = {"jets": _jets_struct(7)}
    comp = ComponentStub("bjets", [jets])
    comps = ComponentsStub([comp])

    merge.write_components(sample=None, components=comps)

    assert isinstance(merge.writer, MemWriter)
    assert merge.writer.num_jets == 7
    assert merge.writer.num_written == 7
    assert merge.writer.dst.name == "merged.h5"


def test_run_groupby_sample_calls_write_components(monkeypatch, tmp_path):
    """Cover run() branch that uses components.groupby_sample()."""
    # Patch H5Writer just in case (unused since we stub write_components)
    monkeypatch.setattr(merging_mod, "H5Writer", MemWriter)

    variables = SimpleNamespace(
        variables=["jets"],
        selectors={},
        combined=lambda: {"jets": None},
        keys=lambda: ["jets"],
    )

    class FakeComponents:
        def __init__(self):
            self.flavours = [Flavours["bjets"]]

        def groupby_sample(self):
            jetsA = {"jets": _jets_struct(2)}
            jetsB = {"jets": _jets_struct(3)}
            compA = ComponentStub("bjets", [jetsA])
            compB = ComponentStub("bjets", [jetsB])
            return [("A", ComponentsStub([compA])), ("B", ComponentsStub([compB]))]

    cfg = SimpleNamespace(
        components=FakeComponents(),
        variables=variables,
        batch_size=100,
        jets_name="jets",
        num_jets_per_output_file=10,
        file_tag="split",
        out_fname=tmp_path / "merged.h5",
        git_hash="deadbeef",
        config={},
        is_test=True,  # force groupby_sample path
        merge_test_samples=False,  # keep per-sample merging
    )
    merge = merging_mod.Merging(cast(PreprocessingConfig, cfg))

    called = []

    def _wc(self, sample, components):  # noqa: ARG001
        called.append((sample, components.num_jets))

    monkeypatch.setattr(merging_mod.Merging, "write_components", _wc)

    merge.run()
    assert called == [("A", 2), ("B", 3)]
