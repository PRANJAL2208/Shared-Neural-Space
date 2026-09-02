"""Latent Neural Interpolation and Continuous Neural Manifold Geometry.

Implements:
1. Spherical Linear Interpolation (SLERP) and Geodesic Traversal on L2-normalized neural hyperspheres.
2. Cross-Brain Semantic Concept Traversal and Monotonic Transition Analysis.
3. Neural Vector Arithmetic (Analogical reasoning in shared latent space).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def slerp(
    p0: np.ndarray | torch.Tensor,
    p1: np.ndarray | torch.Tensor,
    val: float | np.ndarray,
) -> np.ndarray | torch.Tensor:
    """Spherical linear interpolation between two unit vectors p0 and p1.

    Parameters
    ----------
    p0:
        Start vector(s) of shape (D,) or (N, D).
    p1:
        End vector(s) of shape (D,) or (N, D).
    val:
        Interpolation parameter in [0.0, 1.0] (0.0 = p0, 1.0 = p1).

    Returns
    -------
    Interpolated unit vector(s).
    """
    is_torch = isinstance(p0, torch.Tensor)
    if is_torch:
        p0_np = p0.detach().cpu().numpy()
        p1_np = p1.detach().cpu().numpy()
    else:
        p0_np = np.asarray(p0)
        p1_np = np.asarray(p1)

    # Normalize inputs
    p0_norm = p0_np / (np.linalg.norm(p0_np, axis=-1, keepdims=True) + 1e-9)
    p1_norm = p1_np / (np.linalg.norm(p1_np, axis=-1, keepdims=True) + 1e-9)

    dot = np.sum(p0_norm * p1_norm, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)

    # If angle is very small, use linear interpolation
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    val = np.asarray(val)
    if np.all(np.abs(sin_theta) < 1e-6):
        res = (1.0 - val) * p0_norm + val * p1_norm
    else:
        scale0 = np.sin((1.0 - val) * theta) / (sin_theta + 1e-9)
        scale1 = np.sin(val * theta) / (sin_theta + 1e-9)
        res = scale0 * p0_norm + scale1 * p1_norm

    res = res / (np.linalg.norm(res, axis=-1, keepdims=True) + 1e-9)

    if is_torch:
        return torch.from_numpy(res).to(p0.device, dtype=p0.dtype)
    return res


def interpolate_latents(
    z_start: np.ndarray,
    z_end: np.ndarray,
    n_steps: int = 21,
    method: str = "slerp",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a trajectory of interpolated latent vectors between two points.

    Parameters
    ----------
    z_start:
        Start embedding (D,).
    z_end:
        End embedding (D,).
    n_steps:
        Number of interpolation steps along the path.
    method:
        'slerp' (spherical linear) or 'linear'.

    Returns
    -------
    alphas:
        Array of interpolation steps in [0, 1] of shape (n_steps,).
    trajectory:
        Array of interpolated embeddings of shape (n_steps, D).
    """
    alphas = np.linspace(0.0, 1.0, n_steps)
    trajectory = []

    for a in alphas:
        if method == "slerp":
            z_a = slerp(z_start, z_end, a)
        else:
            z_a = (1.0 - a) * z_start + a * z_end
            z_a = z_a / (np.linalg.norm(z_a) + 1e-9)
        trajectory.append(z_a)

    return alphas, np.stack(trajectory, axis=0)


def evaluate_interpolation_monotonicity(
    trajectory: np.ndarray,
    z_start_ref: np.ndarray,
    z_end_ref: np.ndarray,
) -> dict[str, Any]:
    """Evaluate whether similarity along the interpolation path behaves monotonically.

    Parameters
    ----------
    trajectory:
        Interpolated embeddings of shape (n_steps, D).
    z_start_ref:
        Reference embedding for concept A (e.g. CLIP vector or centroid).
    z_end_ref:
        Reference embedding for concept B.

    Returns
    -------
    dict with:
        'sim_to_start': Cosine similarity curve to concept A.
        'sim_to_end': Cosine similarity curve to concept B.
        'start_monotonic_diffs': Differences in sim_to_start (should be <= 0).
        'end_monotonic_diffs': Differences in sim_to_end (should be >= 0).
        'monotonicity_score': Fraction of steps satisfying monotonic transition.
    """
    # Normalize
    traj_norm = trajectory / (np.linalg.norm(trajectory, axis=1, keepdims=True) + 1e-9)
    start_norm = z_start_ref / (np.linalg.norm(z_start_ref) + 1e-9)
    end_norm = z_end_ref / (np.linalg.norm(z_end_ref) + 1e-9)

    sim_start = traj_norm @ start_norm
    sim_end = traj_norm @ end_norm

    diff_start = np.diff(sim_start)
    diff_end = np.diff(sim_end)

    valid_start = np.sum(diff_start <= 1e-4)
    valid_end = np.sum(diff_end >= -1e-4)
    total_steps = 2 * (len(trajectory) - 1)

    monotonicity_score = float((valid_start + valid_end) / total_steps)

    return {
        "sim_to_start": sim_start,
        "sim_to_end": sim_end,
        "monotonicity_score": monotonicity_score,
        "start_monotonic_fraction": float(valid_start / (len(trajectory) - 1)),
        "end_monotonic_fraction": float(valid_end / (len(trajectory) - 1)),
    }


def evaluate_neural_vector_arithmetic(
    concept_embeddings: np.ndarray,
    concept_names: Sequence[str],
    analogies: Sequence[tuple[str, str, str, str]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate neural vector arithmetic (e.g., A - B + C ~ D) on concept centroids.

    Parameters
    ----------
    concept_embeddings:
        Embeddings for all concepts (K, D).
    concept_names:
        Names of all concepts of length K.
    analogies:
        List of tuples (A, B, C, D) representing analogy query: A - B + C = D.
    top_k:
        Top-K neighborhood to check for correct target concept D.

    Returns
    -------
    dict with:
        'top1_accuracy': Fraction of analogies where D is the closest neighbor.
        'topk_accuracy': Fraction of analogies where D is in top_k neighbors.
        'analogy_results': Detailed results per query.
    """
    name_to_idx = {name.lower().strip(): i for i, name in enumerate(concept_names)}
    # Normalize all embeddings
    embeds_norm = concept_embeddings / (np.linalg.norm(concept_embeddings, axis=1, keepdims=True) + 1e-9)

    correct_top1 = 0
    correct_topk = 0
    evaluated = 0
    analogy_results = []

    for a, b, c, target in analogies:
        a_low, b_low, c_low, t_low = a.lower().strip(), b.lower().strip(), c.lower().strip(), target.lower().strip()
        if not (a_low in name_to_idx and b_low in name_to_idx and c_low in name_to_idx and t_low in name_to_idx):
            continue

        idx_a = name_to_idx[a_low]
        idx_b = name_to_idx[b_low]
        idx_c = name_to_idx[c_low]
        idx_t = name_to_idx[t_low]

        v_a = embeds_norm[idx_a]
        v_b = embeds_norm[idx_b]
        v_c = embeds_norm[idx_c]

        # Composite vector: A - B + C
        v_pred = v_a - v_b + v_c
        v_pred = v_pred / (np.linalg.norm(v_pred) + 1e-9)

        # Similarities across all concepts
        sims = embeds_norm @ v_pred
        # Exclude input concepts A, B, C from candidate retrieval
        for excl in [idx_a, idx_b, idx_c]:
            sims[excl] = -np.inf

        ranked_indices = np.argsort(sims)[::-1]
        top_candidates = [concept_names[idx] for idx in ranked_indices[:top_k]]

        is_top1 = (ranked_indices[0] == idx_t)
        is_topk = (idx_t in ranked_indices[:top_k])

        if is_top1:
            correct_top1 += 1
        if is_topk:
            correct_topk += 1
        evaluated += 1

        analogy_results.append({
            "query": f"{a} - {b} + {c}",
            "target": target,
            "predicted_top1": top_candidates[0],
            "top_candidates": top_candidates,
            "is_top1": bool(is_top1),
            "is_topk": bool(is_topk),
            "target_cosine_sim": float(sims[idx_t]),
        })

    top1_acc = float(correct_top1 / evaluated) if evaluated > 0 else 0.0
    topk_acc = float(correct_topk / evaluated) if evaluated > 0 else 0.0

    return {
        "top1_accuracy": top1_acc,
        "topk_accuracy": topk_acc,
        "n_evaluated": evaluated,
        "analogy_results": analogy_results,
    }
