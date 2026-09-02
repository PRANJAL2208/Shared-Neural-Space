"""
THINGS-EEG population dataset loader (ds003825).

Handles:
- EEGDash-mediated recording discovery
- BIDS events.tsv parsing to map trials → THINGS concept IDs/names
- Recording-level ephemeral caching (via CacheManager)
- Channel count inspection (48 × 63-ch, 2 × 128-ch heterogeneity)
- Manifest construction

Usage (in notebook or script)
------------------------------
::

    loader = ThingsPopulationLoader(config)

    # Phase 0: inspect without processing
    info = loader.describe()
    ch_dist = loader.inspect_channels(max_subjects=5)

    # Phase 1+: process one subject
    with CacheManager(config["data_access"]["cache_dir"]).recording("sub-01") as raw_path:
        epochs, meta_list = loader.process_recording("sub-01", raw_path=raw_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.base import TrialMeta
from src.data.eegdash_loader import EEGDashLoader
from src.data.manifest import ManifestBuilder

logger = logging.getLogger(__name__)

DATASET_ID = "ds003825"


class ThingsPopulationLoader:
    """High-level loader for THINGS-EEG ds003825.

    Parameters
    ----------
    config:
        Parsed YAML config dict for ``things_population.yaml``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = config
        ds_cfg = config["dataset"]
        access_cfg = config["data_access"]

        self.loader = EEGDashLoader(
            dataset_id=ds_cfg["eegdash_id"],
            cache_dir=access_cfg["cache_dir"],
            download=True,
        )
        self.manifest_builder = ManifestBuilder(DATASET_ID)

    # ── Dataset-level inspection ──────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """Return top-level metadata from EEGDash."""
        return self.loader.describe()

    def inspect_channels(self, max_subjects: int = 10) -> dict[str, list[str]]:
        """Identify channel count heterogeneity across subjects.

        Returns
        -------
        dict mapping str(n_channels) → [subject_ids]
        """
        dist = self.loader.inspect_channel_counts(max_subjects=max_subjects)
        logger.info("Channel distribution (first %d subjects): %s", max_subjects, dist)
        return dist

    def iter_recordings(
        self, subject: str | None = None
    ):
        """Yield recording metadata dicts."""
        yield from self.loader.iter_recordings(subject=subject)

    # ── Events parsing ────────────────────────────────────────────────────────

    @staticmethod
    def parse_events_tsv(events_tsv_path: str | Path) -> pd.DataFrame:
        """Parse a BIDS events.tsv file for THINGS-EEG ds003825.

        Real column names (verified from live data):
            onset, duration, objectnumber, object, stimname, stim,
            eventnumber, sequencenumber, presentationnumber, istarget, ...

        Returns
        -------
        DataFrame with columns: onset, duration, concept_id, concept_name,
        stimulus_id, event_sample (onset in samples at 1000 Hz).
        """
        df = pd.read_csv(events_tsv_path, sep="\t")
        logger.debug("Events TSV shape: %s, columns: %s", df.shape, df.columns.tolist())

        out = pd.DataFrame()
        # onset is in samples (integer) at 1000 Hz; convert to seconds
        out["event_sample"] = df["onset"].astype(int)      # samples at 1000 Hz
        out["onset_s"]      = df["onset"] / 1000.0         # seconds
        out["duration"]     = df["duration"] / 1000.0      # seconds

        # Concept identity — 'objectnumber' is the THINGS concept index (1-indexed)
        out["concept_id"]   = df.get("objectnumber", pd.Series(dtype=int))
        out["concept_name"] = df.get("object",       pd.Series(dtype=str))

        # Stimulus identity — 'stimname' is the image filename
        out["stimulus_id"]  = df.get("stimname",     df.get("stim", None))

        # Event index for alignment with MNE events array
        out["event_number"] = df.get("eventnumber",  pd.RangeIndex(len(df)))
        out["is_target"]    = df.get("istarget",     0).astype(int)

        return out

    # ── Single-recording processor ────────────────────────────────────────────

    def process_recording(
        self,
        subject: str,
        session: str | None = None,
        run: str | None = None,
        raw_path: str | Path | None = None,
    ) -> tuple["Any", list[TrialMeta]]:
        """Load, preprocess and epoch one recording.

        Returns
        -------
        (epochs, meta_list):
            - epochs: mne.Epochs object
            - meta_list: list of TrialMeta, one per accepted epoch
        """
        import mne  # type: ignore
        from src.preprocessing.pipeline import EEGPreprocessor

        if raw_path is None:
            rec = self.loader.get_recording(subject, session=session, run=run)
            raw_path = rec.get("local_path")
            if raw_path is None:
                raise RuntimeError(
                    f"No local_path available for {subject}/{session}/{run}. "
                    "Ensure download=True and CacheManager is active."
                )

        raw_path = Path(raw_path)
        logger.info("Loading raw: %s", raw_path)

        # MNE can read a directory (BIDS run folder) or a single EEG file
        if raw_path.is_dir():
            # Find the EEG data file within the directory
            candidates = (
                list(raw_path.glob("*.fif"))
                + list(raw_path.glob("*.set"))
                + list(raw_path.glob("*.edf"))
                + list(raw_path.glob("*.bdf"))
            )
            if not candidates:
                raise FileNotFoundError(f"No EEG file found in {raw_path}")
            raw_file = candidates[0]
        else:
            raw_file = raw_path

        raw = mne.io.read_raw(str(raw_file), preload=True, verbose=False)
        logger.info(
            "Loaded: %d channels @ %.0f Hz, %.1f min",
            len(raw.ch_names), raw.info["sfreq"], raw.times[-1] / 60,
        )

        # Preprocess
        pre_cfg = self.cfg.get("preprocessing", {})
        preprocessor = EEGPreprocessor(pre_cfg)
        raw_clean = preprocessor.prepare_raw(raw)

        # Find events (THINGS uses stimulus events)
        events, event_id = mne.events_from_annotations(raw_clean, verbose=False)
        if len(events) == 0:
            logger.warning("No events found — falling back to STIM channel.")
            events = mne.find_events(raw_clean, stim_channel="stim", verbose=False)

        # Epoch
        epoch_cfg = pre_cfg.get("epoch", {})
        epochs = preprocessor.make_epochs(raw_clean, events, event_id, epoch_cfg)

        # Build TrialMeta list
        meta_list = self._build_trial_meta(epochs, subject, session, run)

        return epochs, meta_list

    @staticmethod
    def _build_trial_meta(
        epochs: "Any",
        subject: str,
        session: str | None,
        run: str | None,
    ) -> list[TrialMeta]:
        """Build TrialMeta for each accepted epoch."""
        metas = []
        for i in range(len(epochs)):
            event = epochs.events[i]
            meta = TrialMeta(
                dataset=DATASET_ID,
                subject=subject,
                session=session,
                run=run,
                trial_index=i,
                sampling_rate=epochs.info["sfreq"],
                n_channels=len(epochs.ch_names),
                tmin=epochs.tmin,
                tmax=epochs.tmax,
            )
            meta.global_trial_id = meta.auto_global_id
            # Concept name mapping requires events.tsv — populated externally
            metas.append(meta)
        return metas
