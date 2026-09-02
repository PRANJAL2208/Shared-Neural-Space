"""
Dataset manifest builder and query interface.

A manifest is a Parquet file where each row describes one usable recording
or trial.  It serves as the project's single source of truth for:
- Which recordings exist and where they are on OpenNeuro/NEMAR
- Which have been processed and are available as Zarr arrays
- Which trials were rejected and why

Schema (one row = one recording):
    dataset          str   e.g. "ds003825"
    subject          str   e.g. "sub-01"
    session          str | None
    run              str | None
    recording_id     str   unique key
    n_channels       int
    sampling_rate    float
    n_trials         int | None   (populated after processing)
    source_uri       str | None
    processed        bool
    zarr_path        str | None
    rejected         bool
    rejection_reason str | None

Usage
-----
::

    builder = ManifestBuilder(dataset_id="ds003825")
    builder.add_recording(subject="sub-01", n_channels=63, ...)
    builder.save("./artifacts/manifests/ds003825_manifest.parquet")

    manifest = Manifest.load("./artifacts/manifests/ds003825_manifest.parquet")
    train_recs = manifest.query("processed == True and n_channels == 63")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Columns in their canonical order
_COLUMNS = [
    "dataset", "subject", "session", "run", "recording_id",
    "n_channels", "sampling_rate", "n_trials",
    "source_uri", "processed", "zarr_path",
    "rejected", "rejection_reason",
]


class ManifestBuilder:
    """Incrementally builds a dataset manifest.

    Parameters
    ----------
    dataset_id:
        Dataset identifier (e.g. ``"ds003825"``).
    """

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self._rows: list[dict[str, Any]] = []

    def add_recording(
        self,
        subject: str,
        session: str | None = None,
        run: str | None = None,
        n_channels: int | None = None,
        sampling_rate: float | None = None,
        n_trials: int | None = None,
        source_uri: str | None = None,
        processed: bool = False,
        zarr_path: str | None = None,
        rejected: bool = False,
        rejection_reason: str | None = None,
        **extra: Any,
    ) -> None:
        """Add one recording row to the manifest."""
        rec_id = _make_recording_id(self.dataset_id, subject, session, run)
        row: dict[str, Any] = {
            "dataset": self.dataset_id,
            "subject": subject,
            "session": session,
            "run": run,
            "recording_id": rec_id,
            "n_channels": n_channels,
            "sampling_rate": sampling_rate,
            "n_trials": n_trials,
            "source_uri": source_uri,
            "processed": processed,
            "zarr_path": zarr_path,
            "rejected": rejected,
            "rejection_reason": rejection_reason,
        }
        row.update(extra)
        self._rows.append(row)

    def mark_processed(
        self,
        subject: str,
        session: str | None = None,
        run: str | None = None,
        n_trials: int | None = None,
        zarr_path: str | None = None,
    ) -> None:
        """Update an existing row to reflect successful processing."""
        rec_id = _make_recording_id(self.dataset_id, subject, session, run)
        for row in self._rows:
            if row["recording_id"] == rec_id:
                row["processed"] = True
                if n_trials is not None:
                    row["n_trials"] = n_trials
                if zarr_path is not None:
                    row["zarr_path"] = zarr_path
                return
        logger.warning("Recording %s not found in manifest — adding as new.", rec_id)
        self.add_recording(
            subject=subject, session=session, run=run,
            n_trials=n_trials, zarr_path=zarr_path, processed=True,
        )

    def mark_rejected(
        self,
        subject: str,
        session: str | None = None,
        run: str | None = None,
        reason: str = "unspecified",
    ) -> None:
        """Mark a recording as rejected (e.g. bad data quality)."""
        rec_id = _make_recording_id(self.dataset_id, subject, session, run)
        for row in self._rows:
            if row["recording_id"] == rec_id:
                row["rejected"] = True
                row["rejection_reason"] = reason
                return
        self.add_recording(
            subject=subject, session=session, run=run,
            rejected=True, rejection_reason=reason,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the manifest as a Pandas DataFrame."""
        df = pd.DataFrame(self._rows)
        # Ensure all canonical columns are present
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[_COLUMNS]

    def save(self, path: str | Path) -> Path:
        """Save the manifest to a Parquet file.

        Parameters
        ----------
        path:
            Destination ``.parquet`` file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        logger.info("Manifest saved: %s  (%d rows)", path, len(df))
        return path


class Manifest:
    """Query interface for a saved manifest Parquet file."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        """Load a manifest from a Parquet file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        df = pd.read_parquet(path)
        logger.info("Manifest loaded: %s  (%d rows)", path, len(df))
        return cls(df)

    def query(self, expr: str) -> pd.DataFrame:
        """Pandas DataFrame.query wrapper for readable filtering.

        Example
        -------
        ::

            manifest.query("dataset == 'nm000232' and processed == True")
        """
        return self._df.query(expr)

    def processed(self) -> pd.DataFrame:
        """Return only processed, non-rejected recordings."""
        return self._df[(self._df["processed"] == True) & (self._df["rejected"] == False)]

    def subjects_with_n_channels(self, n: int) -> list[str]:
        """Return subjects whose recording has exactly n channels."""
        return self._df[self._df["n_channels"] == n]["subject"].unique().tolist()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"Manifest({len(self)} rows)"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_recording_id(
    dataset: str, subject: str, session: str | None, run: str | None
) -> str:
    parts = [dataset, subject]
    if session:
        parts.append(session)
    if run:
        parts.append(run)
    return "_".join(parts)
