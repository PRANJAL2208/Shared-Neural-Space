"""
Phase 2: Preprocessing Validation for ds003825 sub-01.

Downloads one subject's recording from S3, runs the full preprocessing
pipeline, saves epochs + concept labels to Zarr, deletes raw cache.

Steps
-----
1. Download sidecars only (events.tsv, .vhdr, .vmrk, .json) — fast
2. Parse events.tsv → concept labels
3. Download the large .eeg file
4. Preprocess with MNE: bandpass, resample, average-ref, epoch
5. Apply ChannelHarmonizer (select 63-channel montage)
6. Normalise (subject-wise baseline)
7. Save to Zarr with provenance
8. Delete raw cache via CacheManager
9. Print shape + label summary

Usage
-----
    $env:PYTHONPATH = "."
    python scripts/phase2_preprocess_validate.py --subject sub-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2")

CONFIG_PATH = Path("configs/things_population.yaml")
SIDECAR_EXTS = (".vhdr", ".vmrk", ".json", "_events.tsv", "_events.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="sub-01")
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--out-zarr", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_zarr = args.out_zarr or config["storage"]["epochs_zarr"]
    cache_dir = Path(config["data_access"]["cache_dir"])
    subject = args.subject

    from src.data.eegdash_loader import OpenNeuroLoader
    from src.data.things_population import ThingsPopulationLoader
    from src.data.zarr_store import ZarrEpochStore
    from src.preprocessing.cache_manager import CacheManager
    from src.preprocessing.montage import ChannelHarmonizer
    from src.preprocessing.normalize import SubjectWiseNormalizer
    from src.preprocessing.pipeline import EEGPreprocessor

    loader = OpenNeuroLoader(
        dataset_id=config["dataset"]["eegdash_id"],
        cache_dir=cache_dir,
        download=True,
    )
    pre_cfg = config.get("preprocessing", {})
    preprocessor = EEGPreprocessor(pre_cfg)
    cache_mgr = CacheManager(cache_dir)
    store = ZarrEpochStore(out_zarr)

    # ── Step 1: Download sidecars first (fast) ────────────────────────────────
    logger.info("Downloading sidecars for %s …", subject)
    subject_dir = cache_dir / config["dataset"]["eegdash_id"] / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    all_files = loader.list_subject_files(subject)
    sidecars = [f for f in all_files
                if any(f["filename"].endswith(ext) for ext in SIDECAR_EXTS)]
    eeg_files = [f for f in all_files if f["filename"].endswith(".eeg")
                 or f["filename"].endswith(".fif") or f["filename"].endswith(".edf")]

    import requests, io
    for f in sidecars:
        dest = subject_dir / "eeg" / f["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            logger.info("  sidecar: %s (%.1f kB)", f["filename"], f["size"] / 1024)
            r = requests.get(f"https://s3.amazonaws.com/openneuro.org/{f['key']}", timeout=60)
            dest.write_bytes(r.content)

    # ── Step 2: Parse events.tsv ──────────────────────────────────────────────
    events_tsv = next((subject_dir / "eeg").glob("*_events.tsv"), None)
    if events_tsv is None:
        logger.error("No events.tsv found in %s", subject_dir / 'eeg')
        sys.exit(1)

    events_df = ThingsPopulationLoader.parse_events_tsv(events_tsv)
    logger.info("Events: %d trials, %d unique concepts",
                len(events_df), events_df["concept_id"].nunique())
    logger.info("Sample:\n%s", events_df[["event_sample", "concept_id", "concept_name", "stimulus_id"]].head(4).to_string())

    # ── Step 3–8: Download EEG, preprocess, save, delete ─────────────────────
    with cache_mgr.recording(subject, recording_path=subject_dir):
        # Download the big .eeg file (streaming)
        for f in eeg_files:
            dest = subject_dir / "eeg" / f["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                logger.info("Downloading %s (%.1f MB) …", f["filename"], f["size"] / 1_048_576)
                import requests as req
                with req.get(f"https://s3.amazonaws.com/openneuro.org/{f['key']}",
                             stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                logger.info("  → saved to %s", dest)

        # Load with MNE
        import mne
        vhdr = next((subject_dir / "eeg").glob("*.vhdr"), None)
        if vhdr is None:
            logger.error("No .vhdr found — cannot load recording.")
            sys.exit(1)

        logger.info("Loading raw from %s …", vhdr)
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        logger.info("Raw: %d ch, %.0f Hz, %.1f min",
                    len(raw.ch_names), raw.info["sfreq"], raw.times[-1] / 60)

        # Preprocessing
        raw_clean = preprocessor.prepare_raw(raw)
        logger.info("After preprocessing: %d ch, %.0f Hz",
                    len(raw_clean.ch_names), raw_clean.info["sfreq"])

        # Build MNE events array from events.tsv (onset = sample at original sfreq)
        # After resampling to 250 Hz, scale onset samples
        orig_sfreq = raw.info["sfreq"]  # before resample (1000 Hz)
        new_sfreq  = raw_clean.info["sfreq"]  # after resample (250 Hz)
        scale = new_sfreq / orig_sfreq

        concept_ids = events_df["concept_id"].values.astype(int)
        onsets_resampled = (events_df["event_sample"].values * scale).astype(int)

        # Clip to valid sample range
        max_sample = raw_clean.n_times - 1
        valid = (onsets_resampled >= 0) & (onsets_resampled < max_sample)
        logger.info("Valid events after resampling: %d / %d", valid.sum(), len(valid))

        mne_events = np.column_stack([
            onsets_resampled[valid],
            np.zeros(valid.sum(), dtype=int),
            concept_ids[valid],
        ])
        events_df_valid = events_df[valid].reset_index(drop=True)

        # Epoch
        epoch_cfg = pre_cfg.get("epoch", {})
        tmin = float(epoch_cfg.get("tmin", -0.2))
        tmax = float(epoch_cfg.get("tmax",  0.8))
        baseline = tuple(epoch_cfg.get("baseline", [-0.2, 0.0]))

        # Use all unique concept IDs as event_id dict
        unique_ids = np.unique(concept_ids[valid])
        event_id = {str(c): int(c) for c in unique_ids}

        epochs = mne.Epochs(
            raw_clean, mne_events, event_id=event_id,
            tmin=tmin, tmax=tmax,
            baseline=baseline,
            preload=True, verbose=False,
            on_missing="ignore",
        )
        # Extract labels from events BEFORE del epochs
        trial_labels = epochs.events[:, 2].astype(np.int32)

        # Memory-safe float32 cast: (22248, 63, 251) float64 = 2.62 GB → 1.31 GB
        # Cast in-place first; get_data() copy then costs only 1.31 GB more
        if hasattr(epochs, '_data') and epochs._data is not None:
            epochs._data = epochs._data.astype(np.float32)
        X = epochs.get_data()   # float32, ~1.31 GB
        sfreq_after = epochs.info["sfreq"]
        del epochs              # free immediately — X is all we need

        logger.info("Epochs: %s  (tmin=%.2f, tmax=%.2f, sfreq=%.0f)",
                    X.shape, tmin, tmax, sfreq_after)

        # ChannelHarmonizer: record channel names for provenance
        harmonizer = ChannelHarmonizer(strategy="select_common", n_channels=63)
        harmonizer.fit_from_raw(raw_clean)
        del raw_clean  # free preprocessed raw — X + events are all we need

        # Normalise subject-wise
        normalizer = SubjectWiseNormalizer(axis=-1)
        X_norm = normalizer.fit_transform(X)
        del X  # free un-normalised copy

        # Validate shape
        EEGPreprocessor.validate_epoch_shape(
            X_norm,
            expected_channels=X_norm.shape[1],
            epoch_duration_s=(tmax - tmin),
            expected_hz=new_sfreq,
        )

        # Labels and concept names
        labels = trial_labels
        concept_names = [
            events_df_valid.loc[events_df_valid["concept_id"] == lbl, "concept_name"]
            .iloc[0] if (events_df_valid["concept_id"] == lbl).any() else "unknown"
            for lbl in labels
        ]

        # Save to Zarr
        provenance = {
            "dataset": config["dataset"]["eegdash_id"],
            "subject": subject,
            "n_trials": int(X_norm.shape[0]),
            "n_channels": int(X_norm.shape[1]),
            "n_samples": int(X_norm.shape[2]),
            "sfreq": float(sfreq_after),
            "tmin": tmin,
            "tmax": tmax,
            "n_unique_concepts": int(np.unique(labels).shape[0]),
            "phase": "2_preprocessing_validation",
        }
        store.write_subject(
            subject=subject,
            X=X_norm,
            labels=labels,
            concept_names=concept_names,
            meta=provenance,
        )
        logger.info("Saved to Zarr at: %s/%s", out_zarr, subject)

    # ── Step 9: Summary ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2 Complete — Preprocessing Validation")
    logger.info("=" * 60)
    logger.info("  Epoch shape       : %s", X_norm.shape)
    logger.info("  Sampling rate     : %.0f Hz", sfreq_after)
    logger.info("  Epoch window      : [%.2f s, %.2f s]", tmin, tmax)
    logger.info("  Unique concepts   : %d", np.unique(labels).shape[0])
    logger.info("  Trials accepted   : %d", X_norm.shape[0])
    logger.info("  Zarr store        : %s", out_zarr)
    logger.info("  Raw cache deleted : %s (ephemeral)", subject_dir)
    logger.info("  Disk after cleanup: %.1f MB", cache_mgr.disk_usage_bytes() / 1_048_576)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
