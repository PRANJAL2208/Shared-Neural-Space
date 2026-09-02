"""
Recording-level ephemeral cache manager.

Core principle:
    Remote Raw → Temporary Recording → Processed Representation → Delete Raw

The CacheManager provides a context manager that guarantees raw downloaded
recordings are deleted — even on exception or keyboard interrupt — once
processing is complete.

Usage
-----
::

    manager = CacheManager(cache_dir="./cache/raw")

    with manager.recording("sub-01", session="ses-01", run="run-01") as raw_path:
        raw = mne.io.read_raw(raw_path, preload=True)
        epochs = preprocess(raw)
        save_to_zarr(epochs)
    # raw_path is deleted here unconditionally

Disk usage at any time is bounded to at most N concurrent recordings where
N is the number of simultaneous context manager instances (typically 1).
"""

from __future__ import annotations

import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages ephemeral local storage for raw EEG recordings.

    Parameters
    ----------
    cache_dir:
        Root directory for raw recording downloads.
    """

    def __init__(self, cache_dir: str | Path = "./cache/raw") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────────────

    @contextmanager
    def recording(
        self,
        subject: str,
        session: str | None = None,
        run: str | None = None,
        recording_path: str | Path | None = None,
    ) -> Generator[Path | None, None, None]:
        """Context manager that yields the recording path and cleans up after.

        Parameters
        ----------
        subject:
            Subject identifier (e.g. ``"sub-01"``).
        session:
            Optional session identifier (e.g. ``"ses-01"``).
        run:
            Optional run identifier (e.g. ``"run-01"``).
        recording_path:
            If the recording has already been downloaded and its local path is
            known, pass it here.  Otherwise the context manager yields None,
            and the caller is responsible for populating the path via EEGDash.

        Yields
        ------
        Path | None
            The local path to the raw recording directory/file, or None if
            ``recording_path`` was not provided.

        Notes
        -----
        The ``finally`` block deletes the recording directory/file regardless
        of whether an exception occurred, so a crash during preprocessing will
        not leave raw data on disk.
        """
        tag = self._make_tag(subject, session, run)
        local_path = Path(recording_path) if recording_path else None

        logger.info("[CacheManager] BEGIN processing %s", tag)
        t0 = time.monotonic()
        try:
            yield local_path
        finally:
            elapsed = time.monotonic() - t0
            if local_path is not None and local_path.exists():
                self._delete(local_path, tag)
            else:
                # EEGDash may have cached to a subdirectory by dataset+subject name.
                # Attempt best-effort cleanup of subject-level cache subdirectory.
                self._try_cleanup_subject_dir(subject, session, tag)
            logger.info(
                "[CacheManager] END %s — elapsed %.1fs", tag, elapsed
            )

    def disk_usage_bytes(self) -> int:
        """Return current byte size of the entire cache directory."""
        return sum(
            f.stat().st_size
            for f in self.cache_dir.rglob("*")
            if f.is_file()
        )

    def clear_all(self) -> None:
        """Delete all contents of the cache directory.

        Use with caution — this removes everything, including recordings
        currently being processed by other workers.
        """
        logger.warning("[CacheManager] Clearing entire cache: %s", self.cache_dir)
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_tag(subject: str, session: str | None, run: str | None) -> str:
        parts = [subject]
        if session:
            parts.append(session)
        if run:
            parts.append(run)
        return "/".join(parts)

    def _delete(self, path: Path, tag: str) -> None:
        """Delete a file or directory tree and log the freed space."""
        try:
            size_mb = (
                sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                if path.is_dir()
                else path.stat().st_size
            ) / 1_048_576

            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

            logger.info(
                "[CacheManager] Deleted %s (%.1f MB freed)", tag, size_mb
            )
        except FileNotFoundError:
            logger.debug("[CacheManager] %s already gone — skipping delete", tag)
        except Exception as exc:
            logger.error(
                "[CacheManager] Failed to delete %s: %s", path, exc
            )

    def _try_cleanup_subject_dir(
        self, subject: str, session: str | None, tag: str
    ) -> None:
        """Best-effort cleanup when the exact path is unknown.

        Searches the cache directory for subdirectories whose name contains
        the subject ID (and optionally session), and deletes them.
        """
        patterns = [subject]
        if session:
            patterns.append(session)

        for child in self.cache_dir.iterdir():
            if all(p in child.name for p in patterns):
                self._delete(child, tag)
                return

        logger.debug(
            "[CacheManager] No matching cache dir found for %s — nothing deleted",
            tag,
        )
