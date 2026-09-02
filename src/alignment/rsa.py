"""
Representational Similarity Analysis (RSA) & Geometry Alignment.

Checkpoint B (H2): Tests whether pairwise concept dissimilarity matrices
(RDMs) computed independently per subject are positively correlated and
whether relational concept geometry is preserved across human brains.

If ρ(RDM_A, RDM_B) > 0 consistently, the relative geometry of concepts is
partially shared across brains — even when raw waveforms differ strongly.

References
----------
Kriegeskorte et al. (2008). Representational similarity analysis —
connecting the branches of systems neuroscience. Front. Syst. Neurosci.
Nili et al. (2014). A toolbox for representational similarity analysis.
PLoS Comput. Biol.
"""

from __future__ import annotations

import logging
from typing import Sequence, Literal

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.linalg import orthogonal_procrustes

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
        [n_concepts, n_features]. Flattened per concept before distance.
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
    return rdm.astype(np.float32)


def vectorize_rdm(rdm: np.ndarray) -> np.ndarray:
    """Return the upper triangle of an RDM (excluding diagonal) as a 1-D vector."""
    idx = np.triu_indices(rdm.shape[0], k=1)
    return rdm[idx].astype(np.float32)


def kendall_tau_a(vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[float, float]:
    """Compute Kendall's tau-a correlation between two RDM vectors.

    Tau-a evaluates ranking concordance without inflating ties, matching
    standard RSA benchmarking practice (Nili et al., 2014).
    """
    n = len(vec_a)
    if n < 2:
        return 0.0, 1.0
    # scipy kendalltau with variant='b' or 'c'; manual tau-a: (concordant - discordant) / (n*(n-1)/2)
    # Using scipy.stats.kendalltau variant='b' which is close, or exact calculation
    res = kendalltau(vec_a, vec_b)
    return float(res.statistic), float(res.pvalue if not np.isnan(res.pvalue) else 1.0)


def rdm_correlation(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    method: Literal["spearman", "pearson", "kendall"] = "spearman",
) -> tuple[float, float]:
    """Correlation between two RDMs.

    Parameters
    ----------
    rdm_a, rdm_b:
        Square RDM arrays of the same shape.
    method:
        ``"spearman"`` (rank correlation, default), ``"pearson"``, or ``"kendall"``.

    Returns
    -------
    (statistic, p_value):
        Correlation coefficient and two-tailed p-value.
    """
    assert rdm_a.shape == rdm_b.shape, "RDMs must have the same shape."
    vec_a = vectorize_rdm(rdm_a)
    vec_b = vectorize_rdm(rdm_b)

    if method == "spearman":
        res = spearmanr(vec_a, vec_b)
        return float(res.statistic), float(res.pvalue)
    elif method == "pearson":
        res = pearsonr(vec_a, vec_b)
        return float(res.statistic), float(res.pvalue)
    elif method == "kendall":
        return kendall_tau_a(vec_a, vec_b)
    else:
        raise ValueError(f"Unknown method {method!r}. Choose spearman, pearson, or kendall.")


def pairwise_rdm_correlations(
    rdms: list[np.ndarray],
    subject_ids: list[str] | None = None,
    method: Literal["spearman", "pearson", "kendall"] = "spearman",
) -> dict:
    """Compute all pairwise RDM correlations across a set of subjects.

    Parameters
    ----------
    rdms:
        List of [n_concepts, n_concepts] RDM arrays, one per subject.
    subject_ids:
        Optional labels for subjects (for logging/output).
    method:
        Correlation metric.

    Returns
    -------
    dict with keys:
        ``"rho_matrix"``  — [n_subjects, n_subjects] pairwise correlation matrix
        ``"mean_rho"``    — mean upper-triangle correlation
        ``"all_rhos"``    — flat array of upper-triangle correlation values
        ``"all_pvalues"`` — corresponding p-values
    """
    n = len(rdms)
    rho_matrix = np.zeros((n, n), dtype=np.float32)
    all_rhos = []
    all_pvals = []

    for i in range(n):
        for j in range(i + 1, n):
            rho, p = rdm_correlation(rdms[i], rdms[j], method=method)
            rho_matrix[i, j] = rho
            rho_matrix[j, i] = rho
            all_rhos.append(rho)
            all_pvals.append(p)

    np.fill_diagonal(rho_matrix, 1.0)
    mean_rho = float(np.mean(all_rhos)) if all_rhos else 1.0
    logger.info(
        "RSA pairwise correlations (%s): mean = %.4f (n_pairs=%d)",
        method, mean_rho, len(all_rhos)
    )

    return {
        "rho_matrix": rho_matrix,
        "mean_rho": mean_rho,
        "all_rhos": np.array(all_rhos, dtype=np.float32),
        "all_pvalues": np.array(all_pvals, dtype=np.float32),
        "subject_ids": subject_ids or [str(i) for i in range(n)],
    }


def permutation_rdm_test(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_permutations: int = 5000,
    random_state: int = 42,
    method: Literal["spearman", "pearson", "kendall"] = "spearman",
) -> dict:
    """Label-permutation test for RDM correlation significance (H2).

    Shuffles the concept ordering of RDM_B and recomputes the RDM correlation
    n_permutations times to build an empirical null distribution.

    Parameters
    ----------
    rdm_a, rdm_b:
        Square RDM arrays.
    n_permutations:
        Number of shuffle iterations.
    random_state:
        Seed for reproducibility.
    method:
        Correlation metric.

    Returns
    -------
    dict with ``"rho_observed"``, ``"null_distribution"``, ``"p_value"``.
    """
    rng = np.random.default_rng(random_state)
    rho_obs, _ = rdm_correlation(rdm_a, rdm_b, method=method)
    vec_a = vectorize_rdm(rdm_a)

    null_rhos = np.empty(n_permutations, dtype=np.float32)
    n = rdm_b.shape[0]

    for i in range(n_permutations):
        perm = rng.permutation(n)
        rdm_b_perm = rdm_b[np.ix_(perm, perm)]
        vec_b_perm = vectorize_rdm(rdm_b_perm)

        if method == "spearman":
            res = spearmanr(vec_a, vec_b_perm)
            null_rhos[i] = float(res.statistic)
        elif method == "pearson":
            res = pearsonr(vec_a, vec_b_perm)
            null_rhos[i] = float(res.statistic)
        elif method == "kendall":
            null_rhos[i], _ = kendall_tau_a(vec_a, vec_b_perm)

    # One-tailed p-value: how often does null >= observed?
    p_val = float((np.sum(null_rhos >= rho_obs) + 1) / (n_permutations + 1))

    logger.info(
        "RSA permutation test (%s): obs=%.4f, p=%.4f (n_perm=%d)",
        method, rho_obs, p_val, n_permutations,
    )
    return {
        "rho_observed": float(rho_obs),
        "null_distribution": null_rhos,
        "p_value": p_val,
    }


def time_resolved_rsa(
    erps_a: np.ndarray,
    erps_b: np.ndarray,
    sfreq: float = 250.0,
    window_ms: float = 50.0,
    tmin_s: float = -0.2,
    method: Literal["spearman", "pearson"] = "spearman",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute time-resolved RSA trajectory across sliding temporal windows.

    Parameters
    ----------
    erps_a, erps_b:
        Array of shape [n_concepts, n_channels, n_timepoints].
    sfreq:
        Sampling frequency in Hz.
    window_ms:
        Width of sliding window in ms.
    tmin_s:
        Start time of epoch in seconds.
    method:
        Correlation metric.

    Returns
    -------
    (times_s, correlations, p_values)
    """
    n_concepts, n_ch, n_time = erps_a.shape
    win_samples = max(1, int(window_ms / 1000.0 * sfreq))

    times_list = []
    rhos_list = []
    pvals_list = []

    for t in range(0, n_time - win_samples + 1):
        sl = slice(t, t + win_samples)
        sub_a = erps_a[:, :, sl]
        sub_b = erps_b[:, :, sl]

        rdm_a_t = compute_rdm(sub_a, metric="correlation")
        rdm_b_t = compute_rdm(sub_b, metric="correlation")

        rho, p = rdm_correlation(rdm_a_t, rdm_b_t, method=method)

        center_time = tmin_s + (t + win_samples / 2.0) / sfreq
        times_list.append(center_time)
        rhos_list.append(rho)
        pvals_list.append(p)

    return (
        np.array(times_list, dtype=np.float32),
        np.array(rhos_list, dtype=np.float32),
        np.array(pvals_list, dtype=np.float32),
    )


def compute_mds_embeddings(
    rdm: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Compute Classical / Metric Multidimensional Scaling (MDS) from an RDM.

    Parameters
    ----------
    rdm:
        [n_concepts, n_concepts] dissimilarity matrix.
    n_components:
        Target dimensionality (typically 2 or 3).
    random_state:
        Seed for optimization.

    Returns
    -------
    embeddings of shape [n_concepts, n_components].
    """
    from sklearn.manifold import MDS
    try:
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            init="random",
            random_state=random_state,
            normalized_stress="auto",
        )
    except TypeError:
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
        )
    embeddings = mds.fit_transform(rdm.astype(np.float64))
    return embeddings.astype(np.float32)


def procrustes_alignment(
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Perform Procrustes analysis aligning representation X_b onto X_a.

    Centers, normalizes scale, and finds optimal orthogonal rotation/reflection R.

    Parameters
    ----------
    X_a, X_b:
        Coordinate matrices of shape [n_concepts, n_features].

    Returns
    -------
    (X_a_centered, X_b_aligned, disparity):
        Aligned matrices and normalized sum of squared Euclidean disparities.
    """
    # Center
    mu_a = X_a.mean(axis=0)
    mu_b = X_b.mean(axis=0)
    A = X_a - mu_a
    B = X_b - mu_b

    # Scale to unit norm
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    A = A / (norm_a if norm_a > 0 else 1.0)
    B = B / (norm_b if norm_b > 0 else 1.0)

    # Orthogonal Procrustes
    R, _ = orthogonal_procrustes(B, A)
    B_aligned = B @ R

    disparity = float(np.sum((A - B_aligned) ** 2))
    return A, B_aligned, disparity
