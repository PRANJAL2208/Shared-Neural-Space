"""
OpenNeuro S3 dataset loader.

Replaces EEGDash for our use-case (one-subject-at-a-time ephemeral processing)
because EEGDash downloads the full dataset on first use — incompatible with
the project's ephemeral-cache strategy.

This loader:
- Queries the public OpenNeuro S3 bucket for metadata WITHOUT downloading
- Lists subjects, files, and sizes via S3 XML listing (no AWS credentials)
- Downloads a single subject's recording into the cache dir on demand
- Supports ``download=False`` for dry-run metadata-only inspection

S3 base URL: https://s3.amazonaws.com/openneuro.org/{dataset_id}/

Usage
-----
::

    loader = OpenNeuroLoader("ds003825", cache_dir="./cache/raw", download=False)

    # Phase 0: inspect metadata without downloading
    info = loader.describe()
    subjects = loader.list_subjects()
    files = loader.list_subject_files("sub-01")

    # Phase 1+: download one subject
    local_dir = loader.download_subject("sub-01")
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

S3_BASE = "https://s3.amazonaws.com/openneuro.org"
S3_NS = "http://s3.amazonaws.com/doc/2006-03-01/"
CHUNK_SIZE = 1 << 20  # 1 MB


class OpenNeuroLoader:
    """Single-subject, ephemeral-cache loader for OpenNeuro datasets via S3.

    Parameters
    ----------
    dataset_id:
        OpenNeuro dataset identifier, e.g. ``"ds003825"``.
    cache_dir:
        Root directory for local caching.  Each subject gets a sub-directory.
    download:
        When False, listing and metadata calls work but ``download_subject()``
        is a no-op (raises RuntimeError). Useful for dry-runs.
    session:
        Optional BIDS session filter (not used in listing, applied by caller).
    """

    def __init__(
        self,
        dataset_id: str,
        cache_dir: str | Path = "./cache/raw",
        download: bool = True,
    ) -> None:
        self.dataset_id = dataset_id
        self.cache_dir = Path(cache_dir)
        self.download = download
        self._s3_prefix = f"{dataset_id}/"

    # ── S3 helpers ────────────────────────────────────────────────────────────

    def _s3_list(self, prefix: str, delimiter: str = "/") -> ET.Element:
        """Return the parsed S3 ListBucketResult XML for the given prefix."""
        params = {"prefix": prefix, "delimiter": delimiter}
        url = f"{S3_BASE}?{urlencode(params)}"
        logger.debug("S3 LIST %s", url)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return ET.fromstring(resp.text)

    def _s3_list_all_keys(self, prefix: str) -> list[dict[str, Any]]:
        """Return all S3 keys under a prefix (handles pagination)."""
        keys = []
        marker = ""
        while True:
            params: dict[str, str] = {"prefix": prefix}
            if marker:
                params["marker"] = marker
            url = f"{S3_BASE}?{urlencode(params)}"
            root = ET.fromstring(requests.get(url, timeout=30).text)
            for content in root.findall(f"{{{S3_NS}}}Contents"):
                key = content.findtext(f"{{{S3_NS}}}Key", "")
                size = int(content.findtext(f"{{{S3_NS}}}Size", "0"))
                keys.append({"key": key, "size": size})
            is_truncated = root.findtext(f"{{{S3_NS}}}IsTruncated", "false")
            if is_truncated.lower() != "true":
                break
            # Next page marker = last key
            marker = keys[-1]["key"] if keys else ""
        return keys

    # ── Public metadata API (no downloads) ────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """Return top-level dataset metadata from S3 (no download)."""
        subjects = self.list_subjects()
        info: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "n_subjects": len(subjects),
            "subjects": subjects,
            "s3_prefix": f"s3://openneuro.org/{self.dataset_id}/",
        }

        # Try to fetch dataset_description.json for Name, Authors etc.
        try:
            url = f"{S3_BASE}/{self.dataset_id}/dataset_description.json"
            r = requests.get(url, timeout=10)
            if r.ok:
                desc = r.json()
                info.update({k: v for k, v in desc.items()
                              if k in ("Name", "BIDSVersion", "Authors", "License")})
        except Exception as exc:
            logger.debug("Could not fetch dataset_description.json: %s", exc)

        logger.info("Dataset %s: %d subjects", self.dataset_id, len(subjects))
        return info

    def list_subjects(self) -> list[str]:
        """Return sorted list of subject IDs from S3."""
        root = self._s3_list(self._s3_prefix, delimiter="/")
        subjects = []
        for cp in root.findall(f"{{{S3_NS}}}CommonPrefixes"):
            prefix = cp.findtext(f"{{{S3_NS}}}Prefix", "")
            # e.g. "ds003825/sub-01/" → "sub-01"
            parts = prefix.rstrip("/").split("/")
            if len(parts) >= 2 and parts[-1].startswith("sub-"):
                subjects.append(parts[-1])
        return sorted(subjects)

    def list_subject_files(
        self, subject: str, extension_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """List all files for a subject (no download).

        Parameters
        ----------
        subject:
            Subject ID, e.g. ``"sub-01"``.
        extension_filter:
            If given, only return files ending with this string
            (e.g. ``".vhdr"``, ``".edf"``).

        Returns
        -------
        List of dicts with ``key``, ``size``, ``filename``.
        """
        prefix = f"{self.dataset_id}/{subject}/"
        all_keys = self._s3_list_all_keys(prefix)
        result = []
        for item in all_keys:
            fname = item["key"].split("/")[-1]
            if extension_filter and not fname.endswith(extension_filter):
                continue
            result.append({**item, "filename": fname})
        logger.info(
            "Subject %s: %d files (%.1f MB total)",
            subject, len(result),
            sum(f["size"] for f in result) / 1_048_576,
        )
        return result

    def inspect_channel_counts(self, max_subjects: int = 5) -> dict[str, list[str]]:
        """Inspect channel count heterogeneity by downloading only .json sidecars.

        Reads ``*_eeg.json`` per subject — tiny files (< 1 KB) — to check
        n_channels without touching the large .eeg/.fif files.
        """
        subjects = self.list_subjects()[:max_subjects]
        counts: dict[str, list[str]] = {}
        for subj in subjects:
            files = self.list_subject_files(subj, extension_filter="_eeg.json")
            if not files:
                # fallback: list any .json
                files = self.list_subject_files(subj, extension_filter=".json")
            for f in files[:1]:  # one sidecar per subject is enough
                try:
                    url = f"{S3_BASE}/{f['key']}"
                    data = requests.get(url, timeout=10).json()
                    n_ch = str(data.get("EEGChannelCount", data.get("ChannelCount", "unknown")))
                    counts.setdefault(n_ch, []).append(subj)
                    break
                except Exception as exc:
                    logger.debug("Could not read sidecar for %s: %s", subj, exc)
        return counts

    # ── Download API ──────────────────────────────────────────────────────────

    def download_subject(
        self,
        subject: str,
        overwrite: bool = False,
    ) -> Path:
        """Download a single subject's EEG files to the cache directory.

        Uses streaming downloads so only one recording is in memory at a time.
        The caller (CacheManager) is responsible for deleting the directory.

        Returns
        -------
        Path to the subject's local directory.
        """
        if not self.download:
            raise RuntimeError(
                "download=False — set download=True to fetch recordings."
            )

        subject_dir = self.cache_dir / self.dataset_id / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        files = self.list_subject_files(subject)
        total_bytes = sum(f["size"] for f in files)
        logger.info(
            "Downloading %s: %d files, %.1f MB",
            subject, len(files), total_bytes / 1_048_576,
        )

        for f in files:
            dest = subject_dir / Path(f["key"]).relative_to(
                f"{self.dataset_id}/{subject}"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() and not overwrite:
                logger.debug("Skip (already cached): %s", dest)
                continue

            url = f"{S3_BASE}/{f['key']}"
            logger.info("Downloading %s (%.1f MB)", f["filename"], f["size"] / 1_048_576)

            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        fh.write(chunk)

        logger.info("Downloaded %s → %s", subject, subject_dir)
        return subject_dir

    # ── Compatibility shims for code written against EEGDashLoader ────────────

    def iter_recordings(
        self,
        subject: str | None = None,
        session: str | None = None,
        run: str | None = None,
        task: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield metadata-only dicts (no download) for compatibility."""
        subjects = [subject] if subject else self.list_subjects()
        for subj in subjects:
            files = self.list_subject_files(subj)
            yield {
                "subject": subj,
                "files": files,
                "n_files": len(files),
                "total_bytes": sum(f["size"] for f in files),
                "local_path": None,  # not downloaded
            }

    def get_recording(
        self, subject: str, session: str | None = None,
        run: str | None = None, task: str | None = None,
    ) -> dict[str, Any]:
        """Download one subject and return the local path dict."""
        local_dir = self.download_subject(subject)
        return {"subject": subject, "local_path": str(local_dir)}


# ── Alias: keep old name working ─────────────────────────────────────────────
EEGDashLoader = OpenNeuroLoader
