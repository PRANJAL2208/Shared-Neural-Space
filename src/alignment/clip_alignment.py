"""
Multimodal EEG-CLIP Representation Alignment.

Maps neural latent representations Z_EEG directly into a foundation vision-language
latent space (CLIP / OpenCLIP 512-d) to enable zero-shot image/concept identification
and cross-modal neural retrieval across human brains.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EEGToCLIPProjector(nn.Module):
    """Deep residual neural projector mapping EEG latent embeddings into CLIP space.

    Parameters
    ----------
    input_dim:
        Dimensionality of input EEG latent representation (e.g. 128).
    clip_dim:
        Target CLIP latent dimensionality (e.g. 512 for ViT-B-32).
    hidden_dim:
        Hidden layer dimensionality (e.g. 256).
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 128,
        clip_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
            nn.LayerNorm(clip_dim),
        )

    def forward(self, z_eeg: torch.Tensor) -> torch.Tensor:
        """Project EEG latent into CLIP embedding space with L2 normalization.

        Parameters
        ----------
        z_eeg:
            Tensor of shape [batch_size, input_dim].

        Returns
        -------
        Normalized CLIP-space embedding of shape [batch_size, clip_dim].
        """
        proj = self.net(z_eeg)
        return F.normalize(proj, p=2, dim=-1)


class MultimodalContrastiveLoss(nn.Module):
    """Bidirectional Cross-Modal Contrastive Loss (EEG <-> CLIP).

    Computes symmetric InfoNCE loss between batch EEG embeddings and target CLIP vectors.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_eeg_proj: torch.Tensor,
        z_clip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_eeg_proj:
            Projected EEG embeddings [B, D], L2-normalized.
        z_clip:
            Target CLIP embeddings [B, D], L2-normalized.

        Returns
        -------
        Scalar loss tensor.
        """
        logits = torch.matmul(z_eeg_proj, z_clip.T) / self.temperature
        batch_size = z_eeg_proj.shape[0]
        labels = torch.arange(batch_size, device=z_eeg_proj.device)

        loss_eeg_to_clip = F.cross_entropy(logits, labels)
        loss_clip_to_eeg = F.cross_entropy(logits.T, labels)
        return (loss_eeg_to_clip + loss_clip_to_eeg) / 2.0


def extract_concept_clip_embeddings(
    concept_names: Sequence[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Extract L2-normalized CLIP text embeddings for a list of visual concepts.

    Parameters
    ----------
    concept_names:
        List of concept string names (e.g. ['airplane', 'banana', ...]).
    model_name:
        CLIP backbone architecture.
    pretrained:
        Pretrained weights checkpoint tag.

    Returns
    -------
    ndarray of shape [n_concepts, clip_dim].
    """
    import open_clip

    logger.info("Loading OpenCLIP model (%s / %s) …", model_name, pretrained)
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    prompts = [f"a clear photograph of a {c.replace('_', ' ')}" for c in concept_names]
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, p=2, dim=-1)

    embeddings = text_features.cpu().numpy().astype(np.float32)
    logger.info("Extracted CLIP embeddings for %d concepts, shape=%s", len(concept_names), embeddings.shape)
    return embeddings
