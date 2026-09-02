"""
Permutation-based statistical testing.

Implements:
1. Label-permutation test for scalar statistics (e.g. same-vs-different ERP Δ)
2. Cluster-based permutation test for time-resolved similarity curves
   (avoids multiple-comparisons over hundreds of time points)

All tests are one-tailed by default (H1: statistic > null).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def permutation_test_scalar(
    observed_stat: float,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    data: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42,
    tail: str = "greater",
) -> dict:
    """Label-permutation significance test for a scalar statistic.

    Shuffles ``labels`` and recomputes ``stat_fn(data, labels_shuffled)``
    to construct a null distribution.

    Parameters
    ----------
    observed_stat:
        The test statistic computed on the real labels.
    stat_fn:
        Callable(data, labels) → float.
    data:
        Input data passed unchanged to stat_fn.
    labels:
        Labels that are shuffled to build the null.
    n_permutations:
        Number of shuffle iterations.
    random_state:
        Seed for reproducibility.
    tail:
        ``"greater"`` (H1: obs > null) or ``"less"`` or ``"two-sided"``.

    Returns
    -------
    dict: ``"observed"``, ``"null_distribution"``, ``"p_value"``, ``"n_permutations"``
    """
    rng = np.random.default_rng(random_state)
    null_stats = np.empty(n_permutations)

    for i in range(n_permutations):
        perm_labels = rng.permutation(labels)
        null_stats[i] = stat_fn(data, perm_labels)

    if tail == "greater":
        p = (np.sum(null_stats >= observed_stat) + 1) / (n_permutations + 1)
    elif tail == "less":
        p = (np.sum(null_stats <= observed_stat) + 1) / (n_permutations + 1)
    else:  # two-sided
        p = (np.sum(np.abs(null_stats) >= abs(observed_stat)) + 1) / (n_permutations + 1)

    logger.info(
        "Permutation test: stat=%.4f, p=%.4f (%s, n_perm=%d)",
        observed_stat, p, tail, n_permutations,
    )
    return {
        "observed": observed_stat,
        "null_distribution": null_stats,
        "p_value": float(p),
        "n_permutations": n_permutations,
        "tail": tail,
    }


def cluster_permutation_test_1d(
    observed_curve: np.ndarray,
    null_curves: np.ndarray,
    threshold: float = 0.05,
    tail: str = "greater",
) -> dict:
    """Cluster-based permutation test for time-resolved statistics (H1 temporal).

    Used for time-resolved cross-subject similarity to avoid inflated Type-I
    error across hundreds of time points.

    Parameters
    ----------
    observed_curve:
        Array [n_timepoints] of the observed statistic at each time point.
    null_curves:
        Array [n_permutations, n_timepoints] of null statistics.
    threshold:
        Per-time-point significance threshold (uncorrected) for initial
        cluster formation.  Default 0.05 (one-tailed vs null percentile).
    tail:
        ``"greater"`` — tests where observed exceeds null.

    Returns
    -------
    dict: ``"cluster_times"``, ``"cluster_stats"``, ``"p_values"``, ``"significant_mask"``
    """
    n_times = len(observed_curve)
    n_perm = null_curves.shape[0]

    # --- Step 1: per-time-point threshold mask ---
    if tail == "greater":
        thresholds = np.percentile(null_curves, 100 * (1 - threshold), axis=0)
        initial_mask = observed_curve > thresholds
    else:
        thresholds = np.percentile(null_curves, 100 * threshold, axis=0)
        initial_mask = observed_curve < thresholds

    # --- Step 2: find observed clusters ---
    obs_clusters = _find_clusters_1d(initial_mask)
    obs_cluster_stats = [
        float(observed_curve[start:end].sum())
        for start, end in obs_clusters
    ]

    # --- Step 3: null cluster-mass distribution ---
    null_max_stats = []
    for perm_i in range(n_perm):
        null_curve = null_curves[perm_i]
        if tail == "greater":
            perm_mask = null_curve > thresholds
        else:
            perm_mask = null_curve < thresholds
        perm_clusters = _find_clusters_1d(perm_mask)
        if perm_clusters:
            null_max_stats.append(
                max(null_curve[s:e].sum() for s, e in perm_clusters)
            )
        else:
            null_max_stats.append(0.0)

    null_max_arr = np.array(null_max_stats)

    # --- Step 4: cluster p-values ---
    cluster_pvals = [
        float((np.sum(null_max_arr >= cs) + 1) / (n_perm + 1))
        for cs in obs_cluster_stats
    ]

    # --- Step 5: significant mask ---
    sig_mask = np.zeros(n_times, dtype=bool)
    for (start, end), pval in zip(obs_clusters, cluster_pvals):
        if pval < threshold:
            sig_mask[start:end] = True

    logger.info(
        "Cluster permutation: %d clusters, %d significant time points.",
        len(obs_clusters), sig_mask.sum(),
    )
    return {
        "cluster_times": obs_clusters,
        "cluster_stats": obs_cluster_stats,
        "p_values": cluster_pvals,
        "significant_mask": sig_mask,
        "null_max_distribution": null_max_arr,
    }


def _find_clusters_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for contiguous True runs in a 1-D mask."""
    clusters = []
    in_cluster = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_cluster:
            in_cluster = True
            start = i
        elif not val and in_cluster:
            in_cluster = False
            clusters.append((start, i))
    if in_cluster:
        clusters.append((start, len(mask)))
    return clusters
