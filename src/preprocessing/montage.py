"""
Channel montage harmonisation across subjects and datasets.

Problem: ds003825 contains 48 subjects with a 63-channel montage and 2 subjects
with a 128-channel montage.  Raw EEG arrays cannot be stacked across subjects
without resolving this heterogeneity first.

Two strategies are supported:

1. ``select_common`` (default / MVP):
   Restrict every subject to the intersection of channels present across all
   *training* subjects.  No interpolation — scientifically clean and safe for
   the initial study.

2. ``interpolate`` (exploratory / later phases):
   Project 128-channel subjects to the 63-channel common montage by dropping
   extra channels or interpolating missing ones via spherical splines (MNE).
   This must be documented explicitly as exploratory in any paper.

Usage
-----
::

    harmonizer = ChannelHarmonizer(strategy="select_common")
    harmonizer.fit(list_of_mne_raw_objects)          # from training subjects
    raw_test = harmonizer.transform(raw_test)         # apply to test subject
"""

from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class ChannelHarmonizer:
    """Ensures channel layout consistency across subjects.

    Parameters
    ----------
    strategy:
        ``"select_common"`` — keep only the intersection of channel names.
        ``"interpolate"``   — interpolate to a reference montage via MNE
                              spherical splines (requires a montage object).
        ``"none"``          — no harmonisation (use when dataset is known to
                              be homogeneous, e.g. nm000232 all 63-ch).
    reference_channel_names:
        When ``strategy="interpolate"``, specify the target channel list
        explicitly instead of fitting from training subjects.
    """

    def __init__(
        self,
        strategy: Literal["select_common", "interpolate", "none"] = "select_common",
        reference_channel_names: list[str] | None = None,
        n_channels: int | None = None,
    ) -> None:
        self.strategy = strategy
        self.reference_channel_names = reference_channel_names
        self.n_channels_target = n_channels  # optional: enforce fixed count
        self._common_channels: list[str] | None = None
        self._fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, raws: Sequence) -> "ChannelHarmonizer":
        """Determine the common channel set from a collection of MNE Raw objects.

        Parameters
        ----------
        raws:
            Iterable of ``mne.io.BaseRaw`` instances from the *training* subjects.
            The test subject must never be included here.

        Returns
        -------
        self
        """
        if self.strategy == "none":
            self._fitted = True
            return self

        if self.strategy == "interpolate" and self.reference_channel_names is not None:
            self._common_channels = list(self.reference_channel_names)
            self._fitted = True
            logger.info(
                "ChannelHarmonizer fitted to %d reference channels (explicit).",
                len(self._common_channels),
            )
            return self

        # Compute intersection of EEG channel names across all training subjects
        channel_sets = []
        for raw in raws:
            eeg_picks = _pick_eeg_channel_names(raw)
            channel_sets.append(set(eeg_picks))

        if not channel_sets:
            raise ValueError("No Raw objects supplied to ChannelHarmonizer.fit().")

        common = channel_sets[0]
        for cset in channel_sets[1:]:
            common = common & cset

        # Sort to ensure deterministic ordering
        self._common_channels = sorted(common)
        self._fitted = True

        counts = {len(s) for s in channel_sets}
        logger.info(
            "ChannelHarmonizer fitted: %d common channels from subjects with "
            "channel counts %s.",
            len(self._common_channels),
            counts,
        )
        return self

    def fit_from_names(self, channel_names: Sequence[str]) -> "ChannelHarmonizer":
        """Fit directly from a list of channel names (e.g. from config)."""
        self._common_channels = list(channel_names)
        self._fitted = True
        return self

    def fit_from_raw(self, raw) -> "ChannelHarmonizer":
        """Convenience: fit from a single MNE Raw (single-subject MVP).

        In the multi-subject case you should call ``fit(list_of_raws)`` so
        only training subjects define the common montage.  For a quick
        single-subject validation, this method fits to the subject itself.
        """
        return self.fit([raw])

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, raw):
        """Apply harmonisation to a single MNE Raw object (in-place copy).

        Parameters
        ----------
        raw:
            An ``mne.io.BaseRaw`` instance (not modified in-place; a new
            Raw is returned after channel selection / interpolation).

        Returns
        -------
        mne.io.BaseRaw
            Raw with only the harmonised channel set.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "ChannelHarmonizer must be fitted before calling transform(). "
                "Call .fit(raws) or .fit_from_names(names) first."
            )

        if self.strategy == "none":
            return raw

        if self._common_channels is None:
            raise RuntimeError("Common channels undefined after fitting.")

        available = set(_pick_eeg_channel_names(raw))
        missing = set(self._common_channels) - available

        if self.strategy == "select_common":
            if missing:
                # Restrict to whatever subset is available
                channels_to_use = [c for c in self._common_channels if c in available]
                logger.warning(
                    "Subject missing %d channels from common montage; "
                    "restricting to %d available channels.  "
                    "Consider excluding this subject.",
                    len(missing),
                    len(channels_to_use),
                )
            else:
                channels_to_use = self._common_channels

            raw_harm = raw.copy().pick(channels_to_use)
            return raw_harm

        elif self.strategy == "interpolate":
            if missing:
                logger.info(
                    "Interpolating %d missing channels via spherical splines.", len(missing)
                )
                raw = raw.copy()
                raw.info["bads"] = list(missing)
                raw.interpolate_bads(reset_bads=True)
            return raw.pick(self._common_channels)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_channels(self) -> int | None:
        """Number of harmonised channels, or None if not yet fitted."""
        return len(self._common_channels) if self._common_channels is not None else None

    @property
    def channel_names(self) -> list[str] | None:
        """Ordered list of harmonised channel names."""
        return self._common_channels


# ── Module-level helpers ───────────────────────────────────────────────────────

def _pick_eeg_channel_names(raw) -> list[str]:
    """Return only EEG channel names from a MNE Raw, excluding EOG/EMG/STIM etc."""
    import mne  # type: ignore
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    return [raw.ch_names[i] for i in picks]


def inspect_channel_distribution(raws: Sequence) -> dict[int, list[str]]:
    """Summarise channel count distribution across subjects.

    Parameters
    ----------
    raws:
        Iterable of (subject_id, mne.io.BaseRaw) tuples.

    Returns
    -------
    dict mapping channel_count → list of subject_ids
    """
    dist: dict[int, list[str]] = {}
    for subj_id, raw in raws:
        n = len(_pick_eeg_channel_names(raw))
        dist.setdefault(n, []).append(subj_id)
    return dist
