"""
CLI: preprocess one subject's recording from THINGS-EEG (ds003825).

Usage
-----
    python scripts/preprocess_subject.py \
        --config configs/things_population.yaml \
        --subject sub-01 \
        [--dry-run]         # inspect only; skip processing and delete step
        [--cache-dir ./cache/raw]
        [--out-zarr ./artifacts/features/ds003825_epochs.zarr]

Design
------
- Fetches one recording via EEGDash (downloads if not cached)
- Preprocesses within CacheManager context → deleted on exit
- Saves epochs to Zarr with full provenance
- Updates the dataset manifest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess_subject")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess one THINGS-EEG subject.")
    parser.add_argument("--config", required=True, help="Path to things_population.yaml")
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. sub-01")
    parser.add_argument("--session", default=None, help="Session ID (if multi-session)")
    parser.add_argument("--run", default=None, help="Run ID (if multi-run)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Inspect dataset metadata only; do not download or preprocess."
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Override cache_dir from config (e.g. /content/cache on Colab)"
    )
    parser.add_argument(
        "--out-zarr", default=None,
        help="Override output Zarr path from config"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.cache_dir:
        config["data_access"]["cache_dir"] = args.cache_dir
    if args.out_zarr:
        config["storage"]["epochs_zarr"] = args.out_zarr

    # ── DRY RUN ──────────────────────────────────────────────────────────────
    if args.dry_run:
        _dry_run(config, args.subject)
        return

    # ── FULL PROCESSING ───────────────────────────────────────────────────────
    _process(config, args.subject, args.session, args.run)


def _dry_run(config: dict, subject: str):
    """Inspect dataset and subject metadata without downloading."""
    from src.data.eegdash_loader import EEGDashLoader

    loader = EEGDashLoader(
        dataset_id=config["dataset"]["eegdash_id"],
        cache_dir=config["data_access"]["cache_dir"],
        download=True,   # needed to query remote catalogue; no files downloaded yet
    )

    logger.info("=== DRY RUN: Dataset inspection ===")
    try:
        info = loader.describe()
        for k, v in info.items():
            logger.info("  %-30s %s", k, v)
    except Exception as exc:
        logger.warning("describe() failed: %s", exc)

    logger.info("=== Channel distribution (first 5 subjects) ===")
    try:
        dist = loader.inspect_channel_counts(max_subjects=5)
        for n_ch, subj_list in dist.items():
            logger.info("  %s channels: %s", n_ch, subj_list)
    except Exception as exc:
        logger.warning("inspect_channel_counts() failed: %s", exc)

    logger.info("=== Recordings for %s ===", subject)
    try:
        recs = list(loader.iter_recordings(subject=subject))
        for rec in recs[:5]:
            logger.info("  %s", rec)
    except Exception as exc:
        logger.warning("iter_recordings() failed: %s", exc)


def _process(config: dict, subject: str, session: str | None, run: str | None):
    """Download, preprocess, save, delete."""
    import numpy as np
    from src.data.things_population import ThingsPopulationLoader
    from src.data.zarr_store import ZarrEpochStore
    from src.preprocessing.cache_manager import CacheManager

    loader_obj = ThingsPopulationLoader(config)
    cache_mgr = CacheManager(config["data_access"]["cache_dir"])
    zarr_path = config["storage"]["epochs_zarr"]
    store = ZarrEpochStore(zarr_path)

    logger.info("Processing %s (session=%s, run=%s)", subject, session, run)

    # Fetch recording metadata first to get local path
    rec_meta = loader_obj.loader.get_recording(subject, session=session, run=run)
    raw_path = rec_meta.get("local_path")

    with cache_mgr.recording(subject, session=session, run=run, recording_path=raw_path):
        epochs, meta_list = loader_obj.process_recording(
            subject, session=session, run=run, raw_path=raw_path
        )

        X = epochs.get_data().astype("float32")  # [trials, channels, samples]
        # Labels: event codes as placeholder (map to concept IDs in full pipeline)
        labels = epochs.events[:, 2].astype("int32")

        provenance = {
            "dataset": "ds003825",
            "subject": subject,
            "session": session,
            "run": run,
            "n_trials": len(epochs),
            "n_channels": X.shape[1],
            "n_samples": X.shape[2],
            "sampling_rate": float(epochs.info["sfreq"]),
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
        }

        store.write_subject(
            subject=subject,
            X=X,
            labels=labels,
            meta=provenance,
        )
        logger.info(
            "Saved to Zarr: %s / %s — shape %s", zarr_path, subject, X.shape
        )

    logger.info(
        "Done. Disk after cache delete: %.1f MB",
        cache_mgr.disk_usage_bytes() / 1_048_576,
    )


if __name__ == "__main__":
    main()
