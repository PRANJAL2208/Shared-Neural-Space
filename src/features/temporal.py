"""
Time-resolved cross-subject ERP similarity analysis (H1, Stage 3–4).

Computes, at each time point, the cross-subject similarity between EEG
responses to the same versus different semantic concepts.

ΔS(t) = S_same(t) - S_different(t)

where:
    S_same(t)      = mean correlation between subject A and subject B
                     ERP at time t when they viewed the same concept.
    S_different(t) = mean correlation when they viewed different concepts.

A positive ΔS(t) at time t suggests that at that moment the two subjects'
neural responses carry shared concept-related information.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def compute_concept_erps(
    X: np.ndarray,
    concept_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average EEG epochs over trials to produce concept-level ERPs.

    Parameters
    ----------
    X:
        Float array [n_trials, n_channels, n_samples].
    concept_labels:
        Integer array [n_trials] of concept identifiers.

    Returns
    -------
    (erps, unique_concepts):
        erps              — [n_concepts, n_channels, n_samples]
        unique_concepts   — [n_concepts] sorted integer concept IDs
    """
    unique = np.sort(np.unique(concept_labels))
    erps = np.stack([X[concept_labels == c].mean(axis=0) for c in unique], axis=0)
    return erps, unique


def cross_subject_similarity_at_time(
    erp_a: np.ndarray,
    erp_b: np.ndarray,
    concepts_a: np.ndarray,
    concepts_b: np.ndarray,
    t: int,
) -> tuple[float, float]:
    """Compute same-concept vs different-concept similarity at one time point.

    Parameters
    ----------
    erp_a, erp_b:
        ERP arrays [n_concepts, n_channels, n_samples] for subjects A and B.
    concepts_a, concepts_b:
        Sorted concept ID arrays matching erp_a and erp_b rows.
    t:
        Time sample index.

    Returns
    -------
    (S_same, S_different)
    """
    # Find shared concepts
    shared = np.intersect1d(concepts_a, concepts_b)
    if len(shared) < 2:
        return float("nan"), float("nan")

    idx_a = {c: i for i, c in enumerate(concepts_a)}
    idx_b = {c: i for i, c in enumerate(concepts_b)}

    same_sims = []
    diff_sims = []

    for i, ca in enumerate(shared):
        vec_a = erp_a[idx_a[ca], :, t]  # [n_channels]
        for j, cb in enumerate(shared):
            vec_b = erp_b[idx_b[cb], :, t]
            r, _ = pearsonr(vec_a, vec_b)
            if ca == cb:
                same_sims.append(r)
            else:
                diff_sims.append(r)

    s_same = float(np.nanmean(same_sims)) if same_sims else float("nan")
    s_diff = float(np.nanmean(diff_sims)) if diff_sims else float("nan")
    return s_same, s_diff


def time_resolved_delta_s(
    erp_a: np.ndarray,
    erp_b: np.ndarray,
    concepts_a: np.ndarray,
    concepts_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ΔS(t) = S_same(t) - S_different(t) over all time points.

    Parameters
    ----------
    erp_a, erp_b:
        [n_concepts, n_channels, n_samples]
    concepts_a, concepts_b:
        Matching concept ID arrays.

    Returns
    -------
    (delta_s, s_same, s_diff)  — each [n_samples]
    """
    n_samples = erp_a.shape[2]
    s_same = np.empty(n_samples)
    s_diff = np.empty(n_samples)

    for t in range(n_samples):
        s_same[t], s_diff[t] = cross_subject_similarity_at_time(
            erp_a, erp_b, concepts_a, concepts_b, t
        )

    delta_s = s_same - s_diff
    logger.info(
        "ΔS: peak=%.3f at t=%d, mean=%.3f",
        float(np.nanmax(delta_s)),
        int(np.nanargmax(delta_s)),
        float(np.nanmean(delta_s)),
    )
    return delta_s, s_same, s_diff
