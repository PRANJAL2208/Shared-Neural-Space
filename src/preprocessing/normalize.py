"""
Leakage-free epoch normalization.

All normalization statistics (mean, std) are fit exclusively on training
subjects.  The test subject's data is only ever *transformed*, never used
to compute any parameters.

This module provides both:
- Subject-wise z-score (fit per subject on that subject's training data)
- Global-train z-score (fit across all training subjects, applied to test)

For LOSO cross-validation, always use GlobalTrainNormalizer and pass only
the subjects in the current outer-fold training set to ``.fit()``.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class SubjectWiseNormalizer:
    """Z-score normalization fitted independently per subject.

    Fit on a subject's own training trials; transform that same subject's
    test trials.  Safe for within-subject experiments; NOT intended for
    cross-subject generalisation claims.

    Parameters
    ----------
    axis:
        Axis over which mean/std are computed.  Default is ``-1`` (time
        axis for shape [trials, channels, samples]).
    """

    def __init__(self, axis: int = -1) -> None:
        self.axis = axis
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SubjectWiseNormalizer":
        """Compute mean and std from X (shape: [trials, channels, samples])."""
        self._mean = X.mean(axis=self.axis, keepdims=True)
        self._std  = X.std(axis=self.axis, keepdims=True)
        logger.debug("SubjectWiseNormalizer fitted on shape %s", X.shape)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Call .fit() before .transform().")
        std = np.where(self._std == 0, 1.0, self._std)
        return (X - self._mean) / std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class GlobalTrainNormalizer:
    """Z-score normalization fitted across all *training* subjects' epochs.

    This is the correct choice for LOSO / cross-subject experiments.
    The test subject's epochs are never used during fitting.

    Parameters
    ----------
    per_channel:
        If True, compute separate mean/std per EEG channel (recommended).
        If False, compute a single scalar (useful only for quick checks).
    """

    def __init__(self, per_channel: bool = True) -> None:
        self.per_channel = per_channel
        self._mean: np.ndarray | None = None
        self._std:  np.ndarray | None = None

    def fit(self, X_train_list: Sequence[np.ndarray]) -> "GlobalTrainNormalizer":
        """Fit on a list of epoch arrays (one per training subject).

        Parameters
        ----------
        X_train_list:
            List of arrays, each with shape [trials, channels, samples].
            Must NOT include the test subject's data.
        """
        combined = np.concatenate(X_train_list, axis=0)  # [all_trials, ch, t]
        if self.per_channel:
            # mean/std over (trials, samples) → shape [1, channels, 1]
            self._mean = combined.mean(axis=(0, 2), keepdims=True)
            self._std  = combined.std(axis=(0, 2),  keepdims=True)
        else:
            self._mean = combined.mean(keepdims=True)
            self._std  = combined.std(keepdims=True)

        logger.info(
            "GlobalTrainNormalizer fitted on %d subjects / %d total trials.",
            len(X_train_list),
            combined.shape[0],
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted normalization to any array (training or test)."""
        if self._mean is None:
            raise RuntimeError("Call .fit() before .transform().")
        std = np.where(self._std == 0, 1.0, self._std)
        return (X - self._mean) / std

    def fit_transform(
        self, X_train_list: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        """Fit on training subjects and return their normalized arrays."""
        self.fit(X_train_list)
        return [self.transform(X) for X in X_train_list]
