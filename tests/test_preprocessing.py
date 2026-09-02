"""Tests for preprocessing pipeline and normalization."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.normalize import GlobalTrainNormalizer, SubjectWiseNormalizer
from src.preprocessing.pipeline import EEGPreprocessor


# ── Pipeline tests ────────────────────────────────────────────────────────────

class TestEEGPreprocessorValidation:
    """Test the static shape-validation helper."""

    def test_valid_shape_exact(self):
        X = np.zeros((32, 63, 250))
        EEGPreprocessor.validate_epoch_shape(X, expected_channels=63)

    def test_valid_shape_plus_one_sample(self):
        """MNE endpoint-inclusive behaviour may produce 251 samples."""
        X = np.zeros((32, 63, 251))
        EEGPreprocessor.validate_epoch_shape(X, expected_channels=63, epoch_duration_s=1.0)

    def test_valid_shape_minus_one_sample(self):
        X = np.zeros((32, 63, 249))
        EEGPreprocessor.validate_epoch_shape(X, expected_channels=63, epoch_duration_s=1.0)

    def test_wrong_ndim_raises(self):
        X = np.zeros((63, 250))  # 2-D, missing trials axis
        with pytest.raises(AssertionError, match="3-D"):
            EEGPreprocessor.validate_epoch_shape(X)

    def test_wrong_channels_raises(self):
        X = np.zeros((10, 128, 250))
        with pytest.raises(AssertionError, match="channels"):
            EEGPreprocessor.validate_epoch_shape(X, expected_channels=63)

    def test_wrong_samples_raises(self):
        """More than ±1 sample deviation should fail."""
        X = np.zeros((10, 63, 300))  # 300 samples vs expected 250
        with pytest.raises(AssertionError, match="time samples"):
            EEGPreprocessor.validate_epoch_shape(X, expected_channels=63, epoch_duration_s=1.0)

    def test_no_channel_constraint(self):
        """When expected_channels is None, any channel count should pass."""
        X = np.zeros((10, 31, 250))
        EEGPreprocessor.validate_epoch_shape(X, expected_channels=None)


# ── Normalization tests ───────────────────────────────────────────────────────

class TestSubjectWiseNormalizer:
    def _make_data(self, n_trials=20, n_ch=10, n_t=100, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=5.0, scale=2.0, size=(n_trials, n_ch, n_t)).astype(np.float32)

    def test_fit_transform_zero_mean(self):
        X = self._make_data()
        norm = SubjectWiseNormalizer(axis=-1)
        X_norm = norm.fit_transform(X)
        # Mean over time axis should be ~0
        assert np.abs(X_norm.mean(axis=-1)).max() < 1e-5

    def test_transform_before_fit_raises(self):
        X = self._make_data()
        norm = SubjectWiseNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            norm.transform(X)

    def test_constant_channel_handled(self):
        """Zero-std channels should not produce NaN (divide by 1 fallback)."""
        X = np.zeros((10, 5, 50), dtype=np.float32)
        X[:, 0, :] = 1.0  # constant channel
        norm = SubjectWiseNormalizer()
        X_norm = norm.fit_transform(X)
        assert not np.any(np.isnan(X_norm))


class TestGlobalTrainNormalizer:
    def _subjects(self, n=3, seed=0):
        rng = np.random.default_rng(seed)
        return [
            rng.normal(loc=float(i), scale=1.0, size=(20, 8, 50)).astype(np.float32)
            for i in range(n)
        ]

    def test_fit_excludes_test_subject(self):
        """Verify that fit() only uses the training subjects passed to it."""
        subjects = self._subjects(4)
        train = subjects[:3]
        test = subjects[3]

        norm = GlobalTrainNormalizer(per_channel=True)
        norm.fit(train)

        # Mean and std should be derived from train only — if we apply to test,
        # the result is a transformed version of test, not its own statistics.
        X_test_norm = norm.transform(test)
        assert X_test_norm.shape == test.shape
        assert not np.any(np.isnan(X_test_norm))

    def test_per_channel_mean_shape(self):
        subjects = self._subjects(3)
        norm = GlobalTrainNormalizer(per_channel=True)
        norm.fit(subjects)
        assert norm._mean.shape == (1, 8, 1)

    def test_fit_transform_returns_list(self):
        subjects = self._subjects(3)
        norm = GlobalTrainNormalizer()
        result = norm.fit_transform(subjects)
        assert len(result) == 3
        for X_norm in result:
            assert X_norm.shape == subjects[0].shape

    def test_transform_before_fit_raises(self):
        norm = GlobalTrainNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            norm.transform(np.zeros((10, 5, 50)))
