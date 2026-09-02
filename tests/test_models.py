"""Unit tests for neural models and contrastive losses."""

from __future__ import annotations

import pytest
import torch

from src.models import (
    EEGNetEncoder,
    SupConLoss,
    MultiSubjectEEGDataset,
    ConceptBalancedBatchSampler,
)


def test_eegnet_forward_shapes():
    B, C, T = 8, 63, 251
    x = torch.randn(B, C, T)
    model = EEGNetEncoder(n_channels=C, n_samples=T, latent_dim=64, projection_dim=32)

    z_feat, z_proj = model(x)
    assert z_feat.shape == (B, 64)
    assert z_proj.shape == (B, 32)

    # Verify L2 normalization
    norms = torch.norm(z_proj, p=2, dim=1)
    torch.testing.assert_close(norms, torch.ones(B), atol=1e-5, rtol=1e-5)


def test_supcon_loss():
    B, D = 12, 32
    features = torch.randn(B, D)
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    # 3 classes, 4 samples each
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    loss_fn = SupConLoss(temperature=0.07)
    loss = loss_fn(features, labels)

    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_dataset_and_sampler():
    import numpy as np
    N, C, T = 100, 63, 251
    X = np.random.randn(N, C, T).astype(np.float32)
    concepts = np.random.randint(0, 10, size=N)
    subjects = np.random.randint(0, 3, size=N)

    ds = MultiSubjectEEGDataset(X, concepts, subjects)
    assert len(ds) == N

    sampler = ConceptBalancedBatchSampler(
        concepts, n_concepts_per_batch=4, n_samples_per_concept=2, n_batches=5
    )
    assert len(sampler) == 5

    for batch in sampler:
        assert len(batch) == 8
