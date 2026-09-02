"""
Contrastive Loss Functions for Cross-Subject Neural Latent Alignment.

Implements Supervised Contrastive Loss (SupCon / InfoNCE) to pull representations
of the same visual concept together across different human brains while pushing
representations of different concepts apart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. NeurIPS 2020).

    Computes contrastive loss over L2-normalized embeddings given concept labels.
    Pairs with the same concept label (regardless of subject origin) are positives.

    Parameters
    ----------
    temperature:
        Scaling parameter for cosine similarity logits (default: 0.07).
    base_temperature:
        Base temperature scaling for loss magnitude (default: 0.07).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Supervised Contrastive Loss.

        Parameters
        ----------
        features:
            L2-normalized embeddings of shape [batch_size, feature_dim].
        labels:
            Ground-truth concept labels of shape [batch_size].
        mask:
            Optional boolean mask where mask[i, j] = 1 if sample i and j are positives.

        Returns
        -------
        Scalar loss tensor.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        if mask is None:
            mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity matrix: [batch_size, batch_size]
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature,
        )

        # For numerical stability: subtract max value
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        # Compute log-probabilities: log(exp(sim) / sum_neg(exp(sim)))
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Only compute loss for anchors that have at least one positive in the batch
        valid_anchors = mask.sum(1) > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid_anchors]
        return loss.mean()
