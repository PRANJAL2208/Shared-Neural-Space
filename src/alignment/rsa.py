"""
Representational Similarity Analysis (RSA).

Checkpoint B (H2): Tests whether pairwise concept dissimilarity matrices
(RDMs) computed independently per subject are positively correlated.

If ρ(RDM_A, RDM_B) > 0 consistently, the relative geometry of concepts is
partially shared across brains — even when raw waveforms differ strongly.

References
----------
Kriegeskorte et al. (2008). Representational similarity analysis —
connecting the branches of systems neuroscience. Front. Syst. Neurosci.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, permutation_test

logger = logging.getLogger(__name__)


def compute_rdm(
    X: np.ndarray,
    metric: str = "correlation",
) -> np.ndarray:
    """Compute a Representational Dissimilarity Matrix for one subject.

    Parameters
    ----------
    X:
        Mean ERP array of shape [n_concepts, n_channels, n_samples] or
        [n_concepts, n_features].  Flattened per concept before distance.
    metric:
        Distance metric passed to ``scipy.spatial.distance.pdist``.
        ``"correlation"`` gives 1 - Pearson r (standard in RSA).

    Returns
    -------
    ndarray of shape [n_concepts, n_concepts].
    """
    n_concepts = X.shape[0]
    X_flat = X.reshape(n_concepts, -1)  # [n_concepts, features]
    dists = pdist(X_flat, metric=metric)
    rdm = squareform(dists)
    return rdm


def vectorize_rdm(rdm: np.ndarray) -> np.ndarray:
    """Return the upper triangle of an RDM (excluding diagonal) as a 1-D vector."""
    idx = np.triu_indices(rdm.shape[0], k=1)
    return rdm[idx]


def rdm_correlation(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
) -> tuple[float, float]:
    """Spearman rank correlation between two RDMs.

    Parameters
    ----------
    rdm_a, rdm_b:
        Square RDM arrays of the same shape.

    Returns
    -------
    (rho, p_value):
        Spearman ρ and two-tailed p-value.
    """
    assert rdm_a.shape == rdm_b.shape, "RDMs must have the same shape."
    vec_a = vectorize_rdm(rdm_a)
    vec_b = vectorize_rdm(rdm_b)
    result = spearmanr(vec_a, vec_b)
    return float(result.statistic), float(result.pvalue)


def pairwise_rdm_correlations(
    rdms: list[np.ndarray],
    subject_ids: list[str] | None = None,
) -> dict:
    """Compute all pairwise RDM correlations across subjects.

    Parameters
    ----------
    rdms:
        List of [n_concepts, n_concepts] RDM arrays, one per subject.
    subject_ids:
        Optional labels for subjects (for logging/output).

    Returns
    -------
    dict with keys:
        ``"rho_matrix"``  — [n_subjects, n_subjects] pairwise ρ matrix
        ``"mean_rho"``    — mean upper-triangle ρ
        ``"all_rhos"``    — flat array of upper-triangle ρ values
        ``"all_pvalues"`` — corresponding p-values
    """
    n = len(rdms)
    rho_matrix = np.zeros((n, n))
    all_rhos = []
    all_pvals = []

    for i in range(n):
        for j in range(i + 1, n):
            rho, p = rdm_correlation(rdms[i], rdms[j])
            rho_matrix[i, j] = rho
            rho_matrix[j, i] = rho
            all_rhos.append(rho)
            all_pvals.append(p)

    np.fill_diagonal(rho_matrix, 1.0)

    mean_rho = float(np.mean(all_rhos))
    logger.info(
        "RSA pairwise correlations: mean ρ = %.3f (n_pairs=%d)", mean_rho, len(all_rhos)
    )

    return {
        "rho_matrix": rho_matrix,
        "mean_rho": mean_rho,
        "all_rhos": np.array(all_rhos),
        "all_pvalues": np.array(all_pvals),
        "subject_ids": subject_ids or [str(i) for i in range(n)],
    }


def permutation_rdm_test(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict:
    """Label-permutation test for RDM correlation significance (H2).

    Shuffles the concept ordering of RDM_B and recomputes ρ n_permutations
    times to build a null distribution.

    Parameters
    ----------
    rdm_a, rdm_b:
        Square RDM arrays.
    n_permutations:
        Number of shuffle iterations.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict with ``"rho_observed"``, ``"null_distribution"``, ``"p_value"``.
    """
    rng = np.random.default_rng(random_state)
    rho_obs, _ = rdm_correlation(rdm_a, rdm_b)
    vec_a = vectorize_rdm(rdm_a)

    null_rhos = []
    n = rdm_b.shape[0]
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        rdm_b_perm = rdm_b[np.ix_(perm, perm)]
        vec_b_perm = vectorize_rdm(rdm_b_perm)
        null_rhos.append(float(spearmanr(vec_a, vec_b_perm).statistic))

    null_arr = np.array(null_rhos)
    # One-tailed: how often does null ≥ observed?
    p_val = (np.sum(null_arr >= rho_obs) + 1) / (n_permutations + 1)

    logger.info(
        "RSA permutation test: ρ_obs=%.3f, p=%.4f (n_perm=%d)",
        rho_obs, p_val, n_permutations,
    )
    return {
        "rho_observed": rho_obs,
        "null_distribution": null_arr,
        "p_value": p_val,
    }
