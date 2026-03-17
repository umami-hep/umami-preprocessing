"""Unit tests for RWMerge._assign_weights."""

from __future__ import annotations

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
