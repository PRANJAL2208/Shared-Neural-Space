"""Tests for RSA module."""

from __future__ import annotations

import numpy as np
import pytest

from src.alignment.rsa import (
    compute_rdm,
    permutation_rdm_test,
    pairwise_rdm_correlations,
    rdm_correlation,
    vectorize_rdm,
)


def _make_erp(n_concepts=10, n_channels=5, n_samples=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_concepts, n_channels, n_samples)).astype(np.float32)


class TestComputeRDM:
    def test_shape(self):
        X = _make_erp(n_concepts=8)
        rdm = compute_rdm(X)
        assert rdm.shape == (8, 8)

    def test_diagonal_zero(self):
        X = _make_erp(n_concepts=6)
        rdm = compute_rdm(X)
        np.testing.assert_allclose(np.diag(rdm), 0.0, atol=1e-6)

    def test_symmetry(self):
        X = _make_erp(n_concepts=6)
        rdm = compute_rdm(X)
        np.testing.assert_allclose(rdm, rdm.T, atol=1e-6)

    def test_range_correlation_metric(self):
        """correlation distances should be in [0, 2]."""
        X = _make_erp(n_concepts=6)
        rdm = compute_rdm(X, metric="correlation")
        assert rdm.min() >= -1e-6
        assert rdm.max() <= 2 + 1e-6


class TestVectorizeRDM:
    def test_length(self):
        rdm = np.zeros((6, 6))
        vec = vectorize_rdm(rdm)
        expected_len = 6 * (6 - 1) // 2  # upper triangle excl. diagonal
        assert len(vec) == expected_len


class TestRDMCorrelation:
    def test_identical_rdms_return_one(self):
        X = _make_erp()
        rdm = compute_rdm(X)
        rho, p = rdm_correlation(rdm, rdm)
        assert abs(rho - 1.0) < 1e-6

    def test_random_rdms_within_bounds(self):
        rdm_a = compute_rdm(_make_erp(seed=0))
        rdm_b = compute_rdm(_make_erp(seed=1))
        rho, p = rdm_correlation(rdm_a, rdm_b)
        assert -1.0 <= rho <= 1.0
        assert 0.0 <= p <= 1.0

    def test_mismatched_shapes_raise(self):
        rdm_a = np.zeros((5, 5))
        rdm_b = np.zeros((6, 6))
        with pytest.raises(AssertionError):
            rdm_correlation(rdm_a, rdm_b)


class TestPairwiseRDMCorrelations:
    def test_output_structure(self):
        rdms = [compute_rdm(_make_erp(seed=i)) for i in range(4)]
        result = pairwise_rdm_correlations(rdms)
        assert "rho_matrix" in result
        assert result["rho_matrix"].shape == (4, 4)
        assert "mean_rho" in result
        assert "all_rhos" in result

    def test_rho_matrix_diagonal_is_one(self):
        rdms = [compute_rdm(_make_erp(seed=i)) for i in range(3)]
        result = pairwise_rdm_correlations(rdms)
        np.testing.assert_allclose(np.diag(result["rho_matrix"]), 1.0, atol=1e-6)


class TestPermutationRDMTest:
    def test_returns_required_keys(self):
        rdm_a = compute_rdm(_make_erp(seed=0))
        rdm_b = compute_rdm(_make_erp(seed=1))
        result = permutation_rdm_test(rdm_a, rdm_b, n_permutations=50)
        for key in ("rho_observed", "null_distribution", "p_value"):
            assert key in result

    def test_p_value_in_range(self):
        rdm_a = compute_rdm(_make_erp(seed=0))
        rdm_b = compute_rdm(_make_erp(seed=1))
        result = permutation_rdm_test(rdm_a, rdm_b, n_permutations=50)
        assert 0.0 < result["p_value"] <= 1.0

    def test_identical_rdms_low_p(self):
        """Identical RDMs should produce a very low p-value."""
        rdm = compute_rdm(_make_erp(seed=42))
        result = permutation_rdm_test(rdm, rdm, n_permutations=200)
        assert result["p_value"] < 0.05, (
            f"Expected p < 0.05 for identical RDMs, got {result['p_value']}"
        )
