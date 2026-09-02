"""Unit tests for latent neural interpolation and neural vector arithmetic."""

import numpy as np
import pytest
import torch

from src.alignment.interpolation import (
    evaluate_interpolation_monotonicity,
    evaluate_neural_vector_arithmetic,
    interpolate_latents,
    slerp,
)


def test_slerp():
    p0 = np.array([1.0, 0.0])
    p1 = np.array([0.0, 1.0])

    # Midpoint
    mid = slerp(p0, p1, 0.5)
    expected_mid = np.array([np.sqrt(0.5), np.sqrt(0.5)])
    assert np.allclose(mid, expected_mid, atol=1e-4)

    # Endpoints
    assert np.allclose(slerp(p0, p1, 0.0), p0, atol=1e-4)
    assert np.allclose(slerp(p0, p1, 1.0), p1, atol=1e-4)

    # PyTorch tensor support
    p0_t = torch.tensor([1.0, 0.0])
    p1_t = torch.tensor([0.0, 1.0])
    mid_t = slerp(p0_t, p1_t, 0.5)
    assert torch.is_tensor(mid_t)
    assert torch.allclose(mid_t, torch.tensor([np.sqrt(0.5), np.sqrt(0.5)], dtype=torch.float32), atol=1e-4)


def test_interpolate_latents():
    z0 = np.array([1.0, 0.0, 0.0])
    z1 = np.array([0.0, 0.0, 1.0])

    alphas, traj = interpolate_latents(z0, z1, n_steps=11, method="slerp")
    assert len(alphas) == 11
    assert traj.shape == (11, 3)
    # Norms should all be 1
    norms = np.linalg.norm(traj, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_evaluate_interpolation_monotonicity():
    z0 = np.array([1.0, 0.0])
    z1 = np.array([0.0, 1.0])
    _, traj = interpolate_latents(z0, z1, n_steps=21, method="slerp")

    res = evaluate_interpolation_monotonicity(traj, z0, z1)
    assert res["monotonicity_score"] > 0.95
    assert res["start_monotonic_fraction"] > 0.95
    assert res["end_monotonic_fraction"] > 0.95


def test_evaluate_neural_vector_arithmetic():
    # Synthetic semantic embedding space:
    # king - man + woman = queen
    names = ["man", "woman", "king", "queen", "apple", "banana"]
    # 2D features: [Gender (0=male, 1=female), Royalty (0=common, 1=royal)]
    embeds = np.array([
        [0.0, 0.0],  # man
        [1.0, 0.0],  # woman
        [0.0, 1.0],  # king
        [1.0, 1.0],  # queen
        [-1.0, -1.0],  # apple
        [-1.0, -2.0],  # banana
    ])

    analogies = [("king", "man", "woman", "queen")]
    res = evaluate_neural_vector_arithmetic(embeds, names, analogies, top_k=2)

    assert res["top1_accuracy"] == 1.0
    assert res["topk_accuracy"] == 1.0
    assert res["n_evaluated"] == 1
    assert res["analogy_results"][0]["predicted_top1"] == "queen"
