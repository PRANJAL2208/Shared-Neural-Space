"""Perception-to-Imagery neural representation alignment and reinstatement.

Implements Stage 3 Neural State Transfer:
1. Perception vs Mental Imagery Representational Dissimilarity Matrix (RDM) alignment.
2. Neural Reinstatement Index computation (congruent vs incongruent cross-modal correlation).
3. Dynamic time-resolved reinstatement tracking across sensory and cognitive windows.
4. Cross-task decoding matrix (Perception -> Imagery generalization).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_concept_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    concepts: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean neural representations (ERP or latent vectors) per concept.

    Parameters
    ----------
    X:
        Feature matrix (N, D) or 3D temporal array (N, C, T) flattened to (N, C*T).
    labels:
        Integer concept labels of length N.
    concepts:
        Optional sorted list of unique concept IDs. If None, derived from unique labels.

    Returns
    -------
    centroids:
        Array of shape (K, D) containing mean vectors for each concept.
    unique_concepts:
        Array of length K containing the concept IDs.
    """
    if X.ndim > 2:
        X_flat = X.reshape(len(X), -1)
    else:
        X_flat = X

    if concepts is None:
        unique_concepts = np.unique(labels)
    else:
        unique_concepts = np.asarray(concepts)

    centroids = []
    valid_concepts = []
    for c in unique_concepts:
        mask = (labels == c)
        if np.any(mask):
            centroids.append(np.mean(X_flat[mask], axis=0))
            valid_concepts.append(c)

    return np.stack(centroids, axis=0), np.array(valid_concepts)


def compute_reinstatement_index(
    X_perc: np.ndarray,
    labels_perc: np.ndarray,
    X_imag: np.ndarray,
    labels_imag: np.ndarray,
    concepts: list[int] | np.ndarray | None = None,
    metric: str = "correlation",
) -> dict[str, Any]:
    """Compute the Perceptual Reinstatement Index between perception and imagery.

    Measures whether mental imagery of concept c is more similar to physical perception
    of concept c (congruent) than to physical perception of concept d != c (incongruent).

    Parameters
    ----------
    X_perc:
        Perception trials (N_p, D).
    labels_perc:
        Perception concept labels (N_p,).
    X_imag:
        Imagery trials (N_i, D).
    labels_imag:
        Imagery concept labels (N_i,).
    concepts:
        Concepts to include.
    metric:
        Distance metric ('correlation' or 'cosine').

    Returns
    -------
    dict with keys:
        's_congruent': Mean similarity between same concept perception & imagery.
        's_incongruent': Mean similarity between different concepts.
        'delta_s': S_congruent - S_incongruent.
        'reinstatement_index': (S_congruent - S_incongruent) / (|S_congruent| + |S_incongruent|).
        'cross_similarity_matrix': Pairwise similarity matrix (K, K).
        'concepts': Array of concepts evaluated.
    """
    if concepts is None:
        common = sorted(list(set(np.unique(labels_perc)) & set(np.unique(labels_imag))))
    else:
        common = sorted(list(set(concepts) & set(np.unique(labels_perc)) & set(np.unique(labels_imag))))

    if len(common) < 2:
        raise ValueError(f"Need at least 2 common concepts between perception and imagery, found {len(common)}")

    mu_p, _ = compute_concept_centroids(X_perc, labels_perc, common)
    mu_i, _ = compute_concept_centroids(X_imag, labels_imag, common)

    if metric == "correlation":
        dists = cdist(mu_p, mu_i, metric="correlation")
        sim_mat = 1.0 - dists
    else:
        dists = cdist(mu_p, mu_i, metric="cosine")
        sim_mat = 1.0 - dists

    k = len(common)
    diag_mask = np.eye(k, dtype=bool)
    off_diag_mask = ~diag_mask

    s_congruent = float(np.mean(sim_mat[diag_mask]))
    s_incongruent = float(np.mean(sim_mat[off_diag_mask]))
    delta_s = float(s_congruent - s_incongruent)

    denom = abs(s_congruent) + abs(s_incongruent)
    reinstatement_idx = float(delta_s / denom) if denom > 1e-9 else 0.0

    return {
        "s_congruent": s_congruent,
        "s_incongruent": s_incongruent,
        "delta_s": delta_s,
        "reinstatement_index": reinstatement_idx,
        "cross_similarity_matrix": sim_mat,
        "concepts": np.array(common),
    }


def permutation_test_reinstatement(
    X_perc: np.ndarray,
    labels_perc: np.ndarray,
    X_imag: np.ndarray,
    labels_imag: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run non-parametric label permutation test on the reinstatement delta."""
    rng = np.random.default_rng(random_state)
    true_res = compute_reinstatement_index(X_perc, labels_perc, X_imag, labels_imag)
    obs_delta = true_res["delta_s"]

    perm_deltas = np.empty(n_permutations, dtype=np.float64)
    shuffled_labels = labels_imag.copy()

    for i in range(n_permutations):
        shuffled_labels = rng.permutation(shuffled_labels)
        res = compute_reinstatement_index(X_perc, labels_perc, X_imag, shuffled_labels)
        perm_deltas[i] = res["delta_s"]

    p_value = float((np.sum(perm_deltas >= obs_delta) + 1.0) / (n_permutations + 1.0))

    return {
        "observed_delta": obs_delta,
        "p_value": p_value,
        "perm_deltas": perm_deltas,
        "reinstatement_index": true_res["reinstatement_index"],
        "s_congruent": true_res["s_congruent"],
        "s_incongruent": true_res["s_incongruent"],
    }


def time_resolved_reinstatement(
    X_perc_3d: np.ndarray,
    labels_perc: np.ndarray,
    X_imag_3d: np.ndarray,
    labels_imag: np.ndarray,
    times: np.ndarray,
    window_samples: int = 5,
) -> dict[str, Any]:
    """Compute time-resolved neural reinstatement across post-stimulus latencies."""
    n_times = len(times)
    delta_curve = np.empty(n_times, dtype=np.float64)
    s_cong_curve = np.empty(n_times, dtype=np.float64)
    s_incong_curve = np.empty(n_times, dtype=np.float64)

    half_win = window_samples // 2

    for t_idx in range(n_times):
        t_start = max(0, t_idx - half_win)
        t_end = min(n_times, t_idx + half_win + 1)

        X_p_t = X_perc_3d[:, :, t_start:t_end].reshape(len(X_perc_3d), -1)
        X_i_t = X_imag_3d[:, :, t_start:t_end].reshape(len(X_imag_3d), -1)

        res = compute_reinstatement_index(X_p_t, labels_perc, X_i_t, labels_imag)
        delta_curve[t_idx] = res["delta_s"]
        s_cong_curve[t_idx] = res["s_congruent"]
        s_incong_curve[t_idx] = res["s_incongruent"]

    peak_idx = int(np.argmax(delta_curve))
    peak_time = float(times[peak_idx])
    peak_delta = float(delta_curve[peak_idx])

    return {
        "times": times,
        "delta_s_curve": delta_curve,
        "s_congruent_curve": s_cong_curve,
        "s_incongruent_curve": s_incong_curve,
        "peak_time": peak_time,
        "peak_delta": peak_delta,
    }


def evaluate_cross_task_decoding_matrix(
    X_perc: np.ndarray,
    y_perc: np.ndarray,
    X_imag: np.ndarray,
    y_imag: np.ndarray,
    n_splits: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and test across Perception and Imagery modalities."""
    if X_perc.ndim > 2:
        X_perc = X_perc.reshape(len(X_perc), -1)
    if X_imag.ndim > 2:
        X_imag = X_imag.reshape(len(X_imag), -1)

    scaler_p = StandardScaler()
    X_perc_s = scaler_p.fit_transform(X_perc)

    scaler_i = StandardScaler()
    X_imag_s = scaler_i.fit_transform(X_imag)

    # 1. Perception -> Perception (CV)
    clf_p_cv = LogisticRegression(C=1.0, max_iter=500, random_state=random_state)
    skf_p = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_p_to_p = []
    for tr, val in skf_p.split(X_perc_s, y_perc):
        clf_p_cv.fit(X_perc_s[tr], y_perc[tr])
        acc_p_to_p.append(accuracy_score(y_perc[val], clf_p_cv.predict(X_perc_s[val])))
    mean_p_to_p = float(np.mean(acc_p_to_p))

    # 2. Perception -> Imagery (Zero-Shot)
    clf_p_full = LogisticRegression(C=1.0, max_iter=500, random_state=random_state)
    clf_p_full.fit(X_perc_s, y_perc)
    X_imag_p_scale = scaler_p.transform(X_imag)
    acc_p_to_i = float(accuracy_score(y_imag, clf_p_full.predict(X_imag_p_scale)))

    # 3. Imagery -> Imagery (CV)
    clf_i_cv = LogisticRegression(C=1.0, max_iter=500, random_state=random_state)
    skf_i = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_i_to_i = []
    for tr, val in skf_i.split(X_imag_s, y_imag):
        clf_i_cv.fit(X_imag_s[tr], y_imag[tr])
        acc_i_to_i.append(accuracy_score(y_imag[val], clf_i_cv.predict(X_imag_s[val])))
    mean_i_to_i = float(np.mean(acc_i_to_i))

    # 4. Imagery -> Perception (Zero-Shot)
    clf_i_full = LogisticRegression(C=1.0, max_iter=500, random_state=random_state)
    clf_i_full.fit(X_imag_s, y_imag)
    X_perc_i_scale = scaler_i.transform(X_perc)
    acc_i_to_p = float(accuracy_score(y_perc, clf_i_full.predict(X_perc_i_scale)))

    n_classes = len(np.unique(y_perc))
    chance = 1.0 / n_classes

    matrix = np.array([
        [mean_p_to_p, acc_p_to_i],
        [acc_i_to_p, mean_i_to_i],
    ])

    return {
        "p_to_p": mean_p_to_p,
        "p_to_i": acc_p_to_i,
        "i_to_i": mean_i_to_i,
        "i_to_p": acc_i_to_p,
        "transfer_matrix": matrix,
        "chance_level": chance,
        "n_classes": n_classes,
    }
