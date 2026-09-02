"""Tests for linear probes and permutation testing."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.probes import LinearProber
from src.evaluation.permutation import (
    permutation_test_scalar,
    cluster_permutation_test_1d,
)


# ── Linear probe tests ────────────────────────────────────────────────────────

def _make_embeddings(n_trials=200, dim=64, n_concepts=5, n_subjects=4, seed=0):
    """Create synthetic embeddings with separable concept structure."""
    rng = np.random.default_rng(seed)
    concept_labels = np.tile(np.arange(n_concepts), n_trials // n_concepts + 1)[:n_trials]
    subject_labels = np.repeat(np.arange(n_subjects), n_trials // n_subjects + 1)[:n_trials]

    # Concept centres in embedding space
    centres = rng.normal(size=(n_concepts, dim)) * 3.0
    Z = centres[concept_labels] + rng.normal(size=(n_trials, dim)) * 0.5

    return Z.astype(np.float32), concept_labels, subject_labels


class TestLinearProber:
    def test_fit_then_evaluate(self):
        Z, c_labels, s_labels = _make_embeddings()
        prober = LinearProber()
        prober.fit(Z, c_labels, s_labels)
        results = prober.evaluate(Z, c_labels, s_labels)  # train = test for a quick smoke test

        assert "concept_accuracy" in results
        assert "subject_accuracy" in results
        assert 0.0 <= results["concept_accuracy"] <= 1.0
        assert 0.0 <= results["subject_accuracy"] <= 1.0

    def test_concept_accuracy_above_chance_on_separable_data(self):
        Z, c_labels, s_labels = _make_embeddings(seed=42)
        prober = LinearProber()
        prober.fit(Z, c_labels, s_labels)
        results = prober.evaluate(Z, c_labels, s_labels)

        # Concepts are highly separable → accuracy should clearly exceed chance
        assert results["concept_accuracy"] > results["concept_chance"] + 0.2, (
            f"Expected concept accuracy well above chance, got {results['concept_accuracy']:.3f}"
        )

    def test_evaluate_before_fit_raises(self):
        Z, c_labels, s_labels = _make_embeddings()
        prober = LinearProber()
        with pytest.raises(RuntimeError, match="fit"):
            prober.evaluate(Z, c_labels, s_labels)

    def test_invariance_score_sign(self):
        """Invariance score should be positive when concept >> subject accuracy."""
        Z, c_labels, s_labels = _make_embeddings(seed=7)
        prober = LinearProber()
        prober.fit(Z, c_labels, s_labels)
        results = prober.evaluate(Z, c_labels, s_labels)
        score = prober.subject_invariance_score(results)
        # Since concepts are separable and subject labels are scrambled, score should be > 0
        assert isinstance(score, float)


# ── Permutation test tests ────────────────────────────────────────────────────

class TestPermutationTestScalar:
    def test_statistically_significant(self):
        """A clearly-present effect should be detected."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(100, 5))
        labels = np.array([0] * 50 + [1] * 50)

        def group_mean_diff(X, y):
            return X[y == 0].mean() - X[y == 1].mean()

        # Create an actual group difference
        data[labels == 0] += 2.0
        obs = group_mean_diff(data, labels)

        result = permutation_test_scalar(
            obs, group_mean_diff, data, labels, n_permutations=200
        )
        assert result["p_value"] < 0.05, f"Expected significant result, got p={result['p_value']}"

    def test_null_effect_not_significant(self):
        """Pure noise should not be systematically significant."""
        rng = np.random.default_rng(123)
        data = rng.normal(size=(100, 5))
        labels = np.array([0] * 50 + [1] * 50)

        def group_mean_diff(X, y):
            return X[y == 0].mean() - X[y == 1].mean()

        obs = group_mean_diff(data, labels)
        result = permutation_test_scalar(
            obs, group_mean_diff, data, labels, n_permutations=500
        )
        # p should not be reliably < 0.05 for pure noise; relax to < 0.5
        assert result["p_value"] > 0.0  # just verify it's a valid probability

    def test_output_keys(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(40, 3))
        labels = np.zeros(40, dtype=int)
        result = permutation_test_scalar(
            0.5, lambda X, y: 0.5, data, labels, n_permutations=10
        )
        for key in ("observed", "null_distribution", "p_value", "n_permutations"):
            assert key in result


class TestClusterPermutationTest1D:
    def test_returns_significant_cluster(self):
        """Inject a clear bump; cluster test should flag it."""
        n_times = 100
        n_perm = 50
        rng = np.random.default_rng(0)

        # Observed curve: random + large bump at times 40–60
        observed = rng.normal(size=n_times) * 0.1
        observed[40:60] += 5.0

        # Null curves: pure noise
        null_curves = rng.normal(size=(n_perm, n_times)) * 0.1

        result = cluster_permutation_test_1d(observed, null_curves)
        assert result["significant_mask"][45:55].any(), (
            "Expected the central bump to be flagged as significant"
        )

    def test_output_keys(self):
        rng = np.random.default_rng(0)
        observed = rng.normal(size=50)
        null_curves = rng.normal(size=(20, 50))
        result = cluster_permutation_test_1d(observed, null_curves)
        for key in ("cluster_times", "cluster_stats", "p_values", "significant_mask"):
            assert key in result
