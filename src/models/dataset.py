"""
PyTorch Dataset and Concept-Balanced Batch Samplers for Cross-Subject EEG.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class MultiSubjectEEGDataset(Dataset):
    """Dataset holding EEG trials, concept labels, and subject indicators."""

    def __init__(
        self,
        X: np.ndarray,
        concept_labels: np.ndarray,
        subject_labels: np.ndarray,
    ) -> None:
        assert len(X) == len(concept_labels) == len(subject_labels)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.concepts = torch.tensor(concept_labels, dtype=torch.long)
        self.subjects = torch.tensor(subject_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.concepts[idx], self.subjects[idx]


class ConceptBalancedBatchSampler(Sampler):
    """Samples batches guaranteed to contain multiple instances of each sampled concept.

    This ensures positive pairs exist in every contrastive batch.
    """

    def __init__(
        self,
        concept_labels: np.ndarray,
        n_concepts_per_batch: int = 16,
        n_samples_per_concept: int = 4,
        n_batches: int = 50,
    ) -> None:
        self.n_concepts_per_batch = n_concepts_per_batch
        self.n_samples_per_concept = n_samples_per_concept
        self.n_batches = n_batches

        self.concept_to_indices: dict[int, list[int]] = {}
        for idx, c in enumerate(concept_labels):
            c_int = int(c)
            if c_int not in self.concept_to_indices:
                self.concept_to_indices[c_int] = []
            self.concept_to_indices[c_int].append(idx)

        self.unique_concepts = list(self.concept_to_indices.keys())

    def __iter__(self):
        for _ in range(self.n_batches):
            selected_concepts = np.random.choice(
                self.unique_concepts,
                size=min(self.n_concepts_per_batch, len(self.unique_concepts)),
                replace=False,
            )
            batch = []
            for c in selected_concepts:
                indices = self.concept_to_indices[c]
                replace = len(indices) < self.n_samples_per_concept
                chosen = np.random.choice(
                    indices, size=self.n_samples_per_concept, replace=replace
                )
                batch.extend(chosen.tolist())

            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches
