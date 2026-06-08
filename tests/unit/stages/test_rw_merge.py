"""Unit tests for RWMerge._assign_weights."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from upp.stages.rw_merge import RWMerge


def _make_rw_data(n_classes=3, n_bins_per_dim=10, n_dims=2):
    """Create synthetic reweighting data for testing."""
    shape = tuple([n_bins_per_dim] * n_dims)
    rw = {
        "weights": {str(c): np.random.rand(*shape).astype(np.float64) for c in range(n_classes)},
    }
    return rw


class TestAssignWeights:
    """Verify _assign_weights produces correct per-jet weights from histogram lookup."""

    @pytest.mark.parametrize("n_jets", [1, 10, 100, 1000])
    @pytest.mark.parametrize("n_dims", [1, 2, 3])
    def test_correct_lookup(self, n_jets, n_dims):
        """Each jet's weight should match the histogram value at its bin and class."""
        n_classes = 4
        n_bins = 8
        rw = _make_rw_data(n_classes=n_classes, n_bins_per_dim=n_bins, n_dims=n_dims)
        bins = np.random.randint(0, n_bins, size=(n_dims, n_jets))
        classes = np.random.randint(0, n_classes, size=n_jets)

        result = RWMerge._assign_weights(rw, bins, classes)

        # Verify each weight matches the histogram value
        for i in range(n_jets):
            expected = rw["weights"][str(classes[i])][tuple(bins[:, i])]
            assert result[i] == expected, f"Mismatch at jet {i}"

    def test_single_class(self):
        rw = _make_rw_data(n_classes=1, n_bins_per_dim=5, n_dims=2)
        bins = np.random.randint(0, 5, size=(2, 50))
        classes = np.zeros(50, dtype=int)

        result = RWMerge._assign_weights(rw, bins, classes)

        for i in range(50):
            assert result[i] == rw["weights"]["0"][tuple(bins[:, i])]

    def test_empty_input(self):
        rw = _make_rw_data(n_classes=3, n_bins_per_dim=5, n_dims=2)
        bins = np.zeros((2, 0), dtype=int)
        classes = np.zeros(0, dtype=int)

        result = RWMerge._assign_weights(rw, bins, classes)
        assert result.shape == (0,)


class TestRWMergeInit:
    """Cover RWMerge.__init__ assertion for outfile_idx_range (line 26)."""

    def test_non_tuple_idx_range_raises(self):
        config = MagicMock()
        with pytest.raises(AssertionError):
            RWMerge(config, outfile_idx_range=[0, 1])

    def test_wrong_length_tuple_raises(self):
        config = MagicMock()
        with pytest.raises(AssertionError):
            RWMerge(config, outfile_idx_range=(0, 1, 2))


class TestStartMp:
    """Cover start_mp multiprocess branch (lines 314-316)."""

    def test_start_mp_multiprocess(self, monkeypatch):
        called_with: list = []

        class FakePool:
            def __init__(self, n: int) -> None:
                pass

            def __enter__(self) -> FakePool:
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def starmap(self, _fn: object, args_list: list) -> None:
                for args in args_list:
                    called_with.append(args)

        monkeypatch.setattr("upp.stages.rw_merge.Pool", FakePool)

        def fn(x: int) -> int:
            return x

        RWMerge.start_mp(fn, [(1,), (2,), (3,)], n_threads=2)
        assert called_with == [(1,), (2,), (3,)]


class TestGetSampleWeightsException:
    """Cover the except-and-reraise path in get_sample_weights (lines 185-187)."""

    def test_exception_propagates(self, monkeypatch):
        n = 5
        jets = np.zeros(n, dtype=[("x", "f4"), ("class_var", "i4")])
        batch = {"jets": jets}
        weights = {
            "jets": {
                "rw1": {
                    "rw_vars": ["x"],
                    "class_var": "class_var",
                    "bins": [np.linspace(0.0, 1.0, 6)],
                    "weights": {"0": np.ones(5)},
                }
            }
        }

        monkeypatch.setattr(
            "upp.stages.rw_merge.bin_jets",
            lambda *_a, **_k: (np.zeros(5), np.zeros((1, n), dtype=int)),
        )

        def bad_assign(*_a: object, **_k: object) -> None:
            raise ValueError("injected error")

        monkeypatch.setattr(RWMerge, "_assign_weights", staticmethod(bad_assign))

        with pytest.raises(ValueError, match="injected error"):
            RWMerge.get_sample_weights(batch, weights)
