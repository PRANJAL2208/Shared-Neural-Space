"""Unit tests for neural scaling laws and population consensus RDM."""

import numpy as np
import pytest

from src.evaluation.scaling import compute_population_consensus_rdm, fit_neural_scaling_law


def test_fit_neural_scaling_law():
    cohort_sizes = [1, 2, 3, 5, 8]
    # Synthetic power-law accuracy: Acc(N) = 0.80 - 0.50 * N^(-0.5)
    true_accs = [0.80 - 0.50 * (n ** -0.5) for n in cohort_sizes]

    res = fit_neural_scaling_law(cohort_sizes, true_accs)
    assert res["A_inf"] > 0.65
    assert res["beta"] > 0.0
    assert res["gamma"] > 0.0
    assert res["r2"] > 0.90
    assert len(res["extrapolated_sizes"]) > 5
    # Accuracy should be monotonically increasing with N
    assert np.all(np.diff(res["extrapolated_acc"]) >= -1e-4)


def test_compute_population_consensus_rdm():
    rng = np.random.default_rng(42)
    k = 10
    true_rdm = rng.uniform(0.1, 0.9, size=(k, k))
    true_rdm = (true_rdm + true_rdm.T) / 2.0
    np.fill_diagonal(true_rdm, 0.0)

    # 4 subjects with noise
    subject_rdms = [true_rdm + 0.2 * rng.normal(size=(k, k)) for _ in range(4)]
    for r in subject_rdms:
        r[:] = (r + r.T) / 2.0
        np.fill_diagonal(r, 0.0)

    res = compute_population_consensus_rdm(subject_rdms)
    assert res["consensus_rdm"].shape == (k, k)
    assert res["snr_gain"] > 1.5
    assert res["mean_subject_to_consensus_rho"] > 0.5
