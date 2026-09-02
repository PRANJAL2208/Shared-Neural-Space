"""
PyTorch Neural Latent Encoders for Single-Trial EEG.

Implements spatial-temporal convolutional backbones and projection heads
for mapping raw multi-channel EEG into a subject-invariant latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetEncoder(nn.Module):
    """Compact Spatial-Temporal Convolutional EEG Encoder (adapted from Lawhern et al. 2018).

    Parameters
    ----------
    n_channels:
        Number of input EEG channels (e.g. 63).
    n_samples:
        Number of timepoints per trial (e.g. 251 at 250 Hz).
    n_spatial_filters:
        Number of spatial filters per temporal kernel (D in EEGNet).
    n_temporal_filters:
        Number of initial temporal filters (F1 in EEGNet).
    latent_dim:
        Dimensionality of the feature representation before projection.
    projection_dim:
        Dimensionality of the final L2-normalized contrastive embedding (e.g. 128).
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        n_channels: int = 63,
        n_samples: int = 251,
        n_temporal_filters: int = 16,
        n_spatial_filters: int = 2,
        latent_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        F1 = n_temporal_filters
        D = n_spatial_filters
        F2 = F1 * D  # total filters after spatial convolution

        # Block 1: Temporal Conv -> Depthwise Spatial Conv -> BatchNorm -> ELU -> AvgPool
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 33), padding=(0, 16), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(
            F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable Conv -> BatchNorm -> ELU -> AvgPool
        self.separable = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 17), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Calculate flattened feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self._forward_features(dummy)
            flat_dim = x.shape[1]

        # Backbone output projection
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
        )

        # Non-linear Projection Head for Contrastive Learning (SimCLR / InfoNCE style)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, projection_dim),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 1, C, T]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(start_dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation Z (unprojected).

        Parameters
        ----------
        x:
            Input tensor of shape [B, n_channels, n_samples] or [B, 1, n_channels, n_samples].

        Returns
        -------
        Latent tensor of shape [B, latent_dim].
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, C, T]
        feat = self._forward_features(x)
        return self.fc(feat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both latent features and L2-normalized projections.

        Returns
        -------
        (z_features, z_projected_normalized)
        """
        z_feat = self.encode(x)
        proj = self.projection_head(z_feat)
        proj_norm = F.normalize(proj, p=2, dim=1)
        return z_feat, proj_norm
