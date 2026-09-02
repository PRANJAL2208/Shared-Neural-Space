"""Unit tests for Perception-to-Imagery neural reinstatement and cross-task decoding."""

import numpy as np
import pytest

from src.alignment.imagery_alignment import (
    compute_concept_centroids,
    compute_reinstatement_index,
    evaluate_cross_task_decoding_matrix,
    permutation_test_reinstatement,
    time_resolved_reinstatement,
)


def test_compute_concept_centroids():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 10))
    labels = np.array([0, 1, 2] * 10)

    centroids, concepts = compute_concept_centroids(X, labels)
    assert centroids.shape == (3, 10)
    assert np.array_equal(concepts, np.array([0, 1, 2]))


def test_compute_reinstatement_index():
    rng = np.random.default_rng(42)
    # Synthetic ground truth: imagery has shared signal with perception plus noise
    n_concepts = 4
    dim = 20
    true_patterns = rng.normal(size=(n_concepts, dim))

    X_perc, y_perc = [], []
    X_imag, y_imag = [], []
    for c in range(n_concepts):
        # 10 trials per concept
        X_perc.append(true_patterns[c] + 0.3 * rng.normal(size=(10, dim)))
        y_perc.extend([c] * 10)
        X_imag.append(true_patterns[c] + 0.5 * rng.normal(size=(10, dim)))
        y_imag.extend([c] * 10)

    X_perc = np.concatenate(X_perc, axis=0)
    y_perc = np.array(y_perc)
    X_imag = np.concatenate(X_imag, axis=0)
    y_imag = np.array(y_imag)

    res = compute_reinstatement_index(X_perc, y_perc, X_imag, y_imag)
    assert res["s_congruent"] > res["s_incongruent"]
    assert res["delta_s"] > 0
    assert res["reinstatement_index"] > 0
    assert res["cross_similarity_matrix"].shape == (4, 4)


def test_permutation_test_reinstatement():
    rng = np.random.default_rng(42)
    n_concepts = 3
    dim = 15
    true_patterns = rng.normal(size=(n_concepts, dim))

    X_perc = np.concatenate([true_patterns[c] + 0.1 * rng.normal(size=(8, dim)) for c in range(n_concepts)])
    y_perc = np.repeat(np.arange(n_concepts), 8)

    X_imag = np.concatenate([true_patterns[c] + 0.2 * rng.normal(size=(8, dim)) for c in range(n_concepts)])
    y_imag = np.repeat(np.arange(n_concepts), 8)

    perm_res = permutation_test_reinstatement(X_perc, y_perc, X_imag, y_imag, n_permutations=100)
    assert perm_res["p_value"] < 0.05
    assert len(perm_res["perm_deltas"]) == 100


def test_time_resolved_reinstatement():
    rng = np.random.default_rng(42)
    n_trials = 20
    n_channels = 4
    n_times = 25
    times = np.linspace(-0.2, 0.8, n_times)

    X_p = rng.normal(size=(n_trials, n_channels, n_times))
    y_p = np.array([0, 1] * (n_trials // 2))
    X_i = rng.normal(size=(n_trials, n_channels, n_times))
    y_i = np.array([0, 1] * (n_trials // 2))

    res = time_resolved_reinstatement(X_p, y_p, X_i, y_i, times)
    assert len(res["delta_s_curve"]) == n_times
    assert -0.2 <= res["peak_time"] <= 0.8


def test_cross_task_decoding_matrix():
    rng = np.random.default_rng(42)
    n_per_class = 15
    dim = 8
    c0 = rng.normal(loc=-1.0, size=(n_per_class, dim))
    c1 = rng.normal(loc=1.0, size=(n_per_class, dim))
    X_p = np.vstack([c0, c1])
    y_p = np.array([0] * n_per_class + [1] * n_per_class)

    # Imagery with similar direction
    c0_i = rng.normal(loc=-0.8, size=(n_per_class, dim))
    c1_i = rng.normal(loc=0.8, size=(n_per_class, dim))
    X_i = np.vstack([c0_i, c1_i])
    y_i = np.array([0] * n_per_class + [1] * n_per_class)

    mat_res = evaluate_cross_task_decoding_matrix(X_p, y_p, X_i, y_i)
    assert mat_res["p_to_p"] > 0.5
    assert mat_res["p_to_i"] > 0.5
    assert mat_res["transfer_matrix"].shape == (2, 2)
