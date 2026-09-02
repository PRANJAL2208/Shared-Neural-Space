"""
EEG preprocessing pipeline.

All steps are configured from a dataset YAML config dict.  The pipeline is
deliberately conservative: each step that could distort data (notch filtering,
artifact rejection) is optional and controlled via config.

Leakage rule: this module never computes normalization statistics globally.
Normalization parameters are fit on training subjects only (see normalize.py).

Pipeline order
--------------
1. Select EEG channels only
2. Drop / interpolate bad channels (conservative)
3. Bandpass filter (0.5–45 Hz default)
4. Notch filter (conditional — only if config specifies)
5. Re-reference (average reference by default)
6. Resample to target_hz (250 Hz default)
7. Epoch around stimulus events
8. Baseline correction
9. (Optional) amplitude-threshold artifact rejection
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Applies the configured preprocessing pipeline to a single MNE Raw.

    Parameters
    ----------
    config:
        Dictionary parsed from a dataset YAML config (``preprocessing`` key).
    verbose:
        MNE verbosity level.
    """

    def __init__(self, config: dict[str, Any], verbose: str = "WARNING") -> None:
        self.cfg = config
        self.verbose = verbose
        import mne as _mne  # lazy import keeps tests runnable without MNE
        _mne.set_log_level(verbose)

    # ── Main entry points ─────────────────────────────────────────────────────

    def prepare_raw(self, raw: "mne.io.BaseRaw") -> "mne.io.BaseRaw":
        """Run all continuous-signal steps (steps 1–6).

        Returns a *new* Raw object; the input is not modified.

        Parameters
        ----------
        raw:
            Loaded MNE Raw, preloaded into memory.
        """
        import mne  # type: ignore
        raw = raw.copy()

        # 1. Select EEG channels only
        raw.pick_types(eeg=True, stim=False, eog=False, emg=False, exclude="bads")
        logger.info("Channels after EEG pick: %d", len(raw.ch_names))

        # 2. Interpolate bad channels (conservative: only if explicitly flagged in info)
        if raw.info["bads"]:
            logger.info("Interpolating %d flagged bad channels.", len(raw.info["bads"]))
            raw.interpolate_bads(reset_bads=True)

        # 3. Bandpass filter
        bp = self.cfg.get("bandpass_hz", [0.5, 45.0])
        if bp:
            raw.filter(l_freq=float(bp[0]), h_freq=float(bp[1]),
                       method="fir", fir_window="hamming")
            logger.info("Bandpass: %.1f–%.1f Hz", bp[0], bp[1])

        # 4. Notch filter (conditional)
        notch = self.cfg.get("notch_hz", None)
        if notch is not None:
            freqs = [float(notch)] if isinstance(notch, (int, float)) else [float(f) for f in notch]
            raw.notch_filter(freqs=freqs)
            logger.info("Notch filter: %s Hz", freqs)

        # 5. Re-reference
        ref = self.cfg.get("reference", "average")
        if ref == "average":
            raw.set_eeg_reference("average", projection=False)
        elif ref not in (None, "none"):
            raw.set_eeg_reference(ref)
        logger.info("Reference: %s", ref)

        # 6. Resample
        target_hz = float(self.cfg.get("resample_hz", 250))
        if abs(raw.info["sfreq"] - target_hz) > 1.0:
            raw.resample(target_hz)
            logger.info("Resampled to %.0f Hz", target_hz)
        else:
            logger.info("Sampling rate already %.0f Hz — skipping resample", raw.info["sfreq"])

        return raw

    def make_epochs(
        self,
        raw: "mne.io.BaseRaw",
        events: "np.ndarray",
        event_id: dict[str, int],
        epoch_cfg: dict[str, Any] | None = None,
    ) -> "mne.Epochs":
        """Create epochs from a prepared Raw and an events array.

        Parameters
        ----------
        raw:
            Preprocessed (continuous) MNE Raw.
        events:
            MNE-format events array (n_events × 3): [sample, prev_id, event_id].
        event_id:
            Mapping from condition name to integer event code.
        epoch_cfg:
            Override for the ``epoch`` sub-dict in the preprocessing config.
            Falls back to ``self.cfg["epoch"]`` if None.

        Returns
        -------
        mne.Epochs
        """
        import mne  # type: ignore
        cfg = epoch_cfg or self.cfg.get("epoch", {})
        tmin = float(cfg.get("tmin", -0.2))
        tmax = float(cfg.get("tmax",  0.8))
        baseline_cfg = cfg.get("baseline", [-0.2, 0.0])
        baseline = (float(baseline_cfg[0]), float(baseline_cfg[1]))

        reject = None
        reject_ptp = cfg.get("reject_peak_to_peak_uv", None)
        if reject_ptp is not None:
            reject = {"eeg": float(reject_ptp) * 1e-6}  # MNE expects volts

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
            preload=True,
            verbose=False,
        )

        n_dropped = len(events) - len(epochs)
        if n_dropped > 0:
            logger.info(
                "Artifact rejection dropped %d / %d epochs (%.1f%%)",
                n_dropped,
                len(events),
                100 * n_dropped / len(events),
            )

        logger.info(
            "Epochs: %d trials × %d channels × %d samples  (%.1f–%.1f s)",
            len(epochs),
            len(epochs.ch_names),
            epochs.get_data().shape[-1],
            tmin,
            tmax,
        )
        return epochs

    # ── Validation helper ─────────────────────────────────────────────────────

    @staticmethod
    def validate_epoch_shape(
        X: "np.ndarray",
        expected_channels: int | None = None,
        expected_hz: float = 250.0,
        epoch_duration_s: float = 1.0,
    ) -> None:
        """Assert that an epoch array has the expected structure.

        Parameters
        ----------
        X:
            Array of shape (n_trials, n_channels, n_samples).
        expected_channels:
            If given, assert that X.shape[1] matches.
        expected_hz:
            Expected sampling rate (used to compute expected n_samples).
        epoch_duration_s:
            Duration of the epoch in seconds.

        Raises
        ------
        AssertionError
            If any validation fails.
        """
        assert X.ndim == 3, (
            f"Expected 3-D epoch array (trials, channels, samples), got {X.ndim}-D"
        )

        if expected_channels is not None:
            assert X.shape[1] == expected_channels, (
                f"Expected {expected_channels} channels, got {X.shape[1]}"
            )

        # Allow ±1 sample for MNE endpoint-inclusive behaviour
        expected_samples = int(expected_hz * epoch_duration_s)
        assert abs(X.shape[2] - expected_samples) <= 1, (
            f"Expected ~{expected_samples} time samples (±1), got {X.shape[2]}"
        )
