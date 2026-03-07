"""Unit tests for RWMerge._assign_weights vectorized implementation."""

from __future__ import annotations

import time

import numpy as np
import pytest

from upp.stages.rw_merge import RWMerge


def _make_rw_data(n_classes=3, n_bins_per_dim=10, n_dims=2):
    """Create synthetic reweighting data for testing."""
    shape = tuple([n_bins_per_dim] * n_dims)
    rw = {
        "weights": {
            str(c): np.random.rand(*shape).astype(np.float64) for c in range(n_classes)
        },
    }
    return rw


def _assign_weights_original(this_rw, bins, to_dump):
    """Original per-jet loop implementation for comparison."""
    this_weights = np.zeros(to_dump.shape, dtype=float)
    for i in range(this_weights.shape[0]):
        bin_idx = bins[:, i]
        cls = to_dump[i]
        thishist = this_rw["weights"][str(cls)][tuple(bin_idx)]
        this_weights[i] = thishist
    return this_weights


class TestAssignWeightsCorrectness:
    """Verify vectorized _assign_weights matches the original loop."""

    @pytest.mark.parametrize("n_jets", [1, 10, 100, 1000])
    @pytest.mark.parametrize("n_dims", [1, 2, 3])
    def test_matches_original(self, n_jets, n_dims):
        n_classes = 4
        n_bins = 8
        rw = _make_rw_data(n_classes=n_classes, n_bins_per_dim=n_bins, n_dims=n_dims)
        bins = np.random.randint(0, n_bins, size=(n_dims, n_jets))
        classes = np.random.randint(0, n_classes, size=n_jets)

        expected = _assign_weights_original(rw, bins, classes)
        result = RWMerge._assign_weights(rw, bins, classes)

        np.testing.assert_array_equal(result, expected)

    def test_single_class(self):
        rw = _make_rw_data(n_classes=1, n_bins_per_dim=5, n_dims=2)
        bins = np.random.randint(0, 5, size=(2, 50))
        classes = np.zeros(50, dtype=int)

        expected = _assign_weights_original(rw, bins, classes)
        result = RWMerge._assign_weights(rw, bins, classes)
        np.testing.assert_array_equal(result, expected)

    def test_empty_input(self):
        rw = _make_rw_data(n_classes=3, n_bins_per_dim=5, n_dims=2)
        bins = np.zeros((2, 0), dtype=int)
        classes = np.zeros(0, dtype=int)

        result = RWMerge._assign_weights(rw, bins, classes)
        assert result.shape == (0,)


class TestAssignWeightsPerformance:
    """Benchmark vectorized vs original implementation."""

    @pytest.mark.parametrize("n_jets", [10_000, 100_000, 250_000])
    def test_vectorized_faster(self, n_jets):
        n_classes = 5
        n_bins = 15
        n_dims = 2
        rw = _make_rw_data(n_classes=n_classes, n_bins_per_dim=n_bins, n_dims=n_dims)
        bins = np.random.randint(0, n_bins, size=(n_dims, n_jets))
        classes = np.random.randint(0, n_classes, size=n_jets)

        # Warm up
        _assign_weights_original(rw, bins[:, :100], classes[:100])
        RWMerge._assign_weights(rw, bins[:, :100], classes[:100])

        # Time original
        t0 = time.perf_counter()
        expected = _assign_weights_original(rw, bins, classes)
        t_original = time.perf_counter() - t0

        # Time vectorized
        t0 = time.perf_counter()
        result = RWMerge._assign_weights(rw, bins, classes)
        t_vectorized = time.perf_counter() - t0

        np.testing.assert_array_equal(result, expected)

        speedup = t_original / t_vectorized
        print(
            f"\n  n_jets={n_jets:>7,}: "
            f"original={t_original:.4f}s, vectorized={t_vectorized:.4f}s, "
            f"speedup={speedup:.1f}x"
        )
        # Vectorized should be significantly faster for large batches
        if n_jets >= 100_000:
            assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"
