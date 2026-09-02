"""
Subject-identification and concept-identification linear probes (H4).

After training a subject-invariant contrastive encoder, we verify empirically:
    Probe 1 (concept): Z → concept    should be HIGH
    Probe 2 (subject): Z → subject    should be at CHANCE

This provides quantitative evidence that the network learned a subject-
invariant representation rather than merely appearing so in UMAP plots.

Usage
-----
::

    prober = LinearProber(embedding_dim=256)
    prober.fit(Z_train, concept_labels_train, subject_labels_train)
    results = prober.evaluate(Z_test, concept_labels_test, subject_labels_test)
    print(results)
    # {'concept_accuracy': 0.72, 'subject_accuracy': 0.11, 'n_subjects': 10, ...}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class LinearProber:
    """Trains and evaluates two linear probes on frozen embeddings.

    The probes are intentionally simple (logistic regression) so that any
    accuracy comes from information in Z, not from the probe's capacity.

    Parameters
    ----------
    max_iter:
        Max iterations for LogisticRegression solver.
    C:
        Inverse regularization strength.  Larger = less regularization.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state

        self._concept_probe = LogisticRegression(
            max_iter=max_iter, C=C, random_state=random_state,
            solver="lbfgs",
        )
        self._subject_probe = LogisticRegression(
            max_iter=max_iter, C=C, random_state=random_state,
            solver="lbfgs",
        )
        self._concept_enc = LabelEncoder()
        self._subject_enc = LabelEncoder()
        self._fitted = False

    def fit(
        self,
        Z: np.ndarray,
        concept_labels: np.ndarray,
        subject_labels: np.ndarray,
    ) -> "LinearProber":
        """Train both probes on frozen embeddings from training subjects.

        Parameters
        ----------
        Z:
            Float32 embeddings of shape [n_trials, embedding_dim].
        concept_labels:
            Integer or string concept labels of shape [n_trials].
        subject_labels:
            Integer or string subject IDs of shape [n_trials].

        Returns
        -------
        self
        """
        y_concept = self._concept_enc.fit_transform(concept_labels)
        y_subject = self._subject_enc.fit_transform(subject_labels)

        self._concept_probe.fit(Z, y_concept)
        self._subject_probe.fit(Z, y_subject)
        self._fitted = True

        n_concepts = len(self._concept_enc.classes_)
        n_subjects = len(self._subject_enc.classes_)
        logger.info(
            "LinearProber fitted: %d concepts, %d subjects, %d trials.",
            n_concepts, n_subjects, len(Z),
        )
        return self

    def evaluate(
        self,
        Z: np.ndarray,
        concept_labels: np.ndarray,
        subject_labels: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate both probes on held-out embeddings.

        Returns
        -------
        dict with:
            ``"concept_accuracy"``   — fraction correct for concept probe
            ``"subject_accuracy"``   — fraction correct for subject probe
            ``"subject_chance"``     — 1 / n_subjects (random baseline)
            ``"concept_chance"``     — 1 / n_concepts
            ``"n_concepts"``
            ``"n_subjects"``
            ``"n_trials"``
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .evaluate().")

        y_concept = self._concept_enc.transform(concept_labels)
        y_subject = self._subject_enc.transform(subject_labels)

        concept_acc = float(self._concept_probe.score(Z, y_concept))
        subject_acc = float(self._subject_probe.score(Z, y_subject))

        n_concepts = len(self._concept_enc.classes_)
        n_subjects = len(self._subject_enc.classes_)

        results = {
            "concept_accuracy": concept_acc,
            "subject_accuracy": subject_acc,
            "concept_chance": 1.0 / n_concepts,
            "subject_chance": 1.0 / n_subjects,
            "n_concepts": n_concepts,
            "n_subjects": n_subjects,
            "n_trials": len(Z),
        }

        logger.info(
            "Probe results: concept=%.3f (chance=%.3f), subject=%.3f (chance=%.3f)",
            concept_acc, 1.0 / n_concepts,
            subject_acc, 1.0 / n_subjects,
        )
        return results

    def subject_invariance_score(self, results: dict[str, Any]) -> float:
        """Scalar summarising subject invariance.

        A score of 1.0 means concept accuracy is perfectly above chance and
        subject accuracy is exactly at chance.  Used for hyperparameter
        selection.

        Score = (concept_acc - concept_chance) - (subject_acc - subject_chance)
        """
        ca = results["concept_accuracy"]
        cc = results["concept_chance"]
        sa = results["subject_accuracy"]
        sc = results["subject_chance"]
        return (ca - cc) - (sa - sc)
