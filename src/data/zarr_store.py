"""
Zarr-backed epoch and embedding storage.

Stores processed EEG epochs and model embeddings with full per-trial
provenance.  Organised as a Zarr hierarchy:

    store.zarr/
    ├── sub-01/
    │   ├── eeg           float32 array [n_trials, n_channels, n_samples]
    │   ├── labels        int32   array [n_trials]
    │   ├── concept_names str     array [n_trials]
    │   └── .zattrs       {dataset, subject, session, run, tmin, tmax, ...}
    ├── sub-02/ ...

Reads are slice-aware — you can load a subset of trials without reading
the whole array (e.g. ``store["sub-01/eeg"][0:256]``).

Zarr v2/v3 compatibility
------------------------
Zarr v3 removed the ``compressor=`` kwarg (zarr_format=2 API).
This module detects the installed version at runtime and uses the
appropriate codec API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _zarr_version() -> int:
    """Return the major version of the installed zarr package."""
    try:
        import zarr
        return int(zarr.__version__.split(".")[0])
    except Exception:
        return 2


def _import_zarr():
    try:
        import zarr  # type: ignore
        return zarr
    except ImportError as exc:
        raise ImportError("zarr is required: pip install zarr") from exc


class ZarrEpochStore:
    """Append-ready Zarr store for processed EEG epochs.

    Parameters
    ----------
    zarr_path:
        Path to the ``.zarr`` directory (created if absent).
    """

    def __init__(self, zarr_path: str | Path) -> None:
        self.zarr_path = Path(zarr_path)
        self._zarr = _import_zarr()
        self._ver = _zarr_version()
        self._root = self._zarr.open(str(self.zarr_path), mode="a")
        logger.debug("ZarrEpochStore: zarr v%d at %s", self._ver, self.zarr_path)

    # ── Internal: version-safe array write ────────────────────────────────────

    def _write_array(self, data: np.ndarray, path: str, **extra) -> None:
        """Write a numpy array to the store using the correct zarr API."""
        if self._ver >= 3:
            # zarr v3 codec pipeline: ArrayBytesCodec then bytes-to-bytes codecs
            from zarr.codecs import BytesCodec, BloscCodec  # type: ignore
            self._zarr.array(
                data,
                store=self._root.store,
                path=path,
                codecs=[BytesCodec(), BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")],
                overwrite=True,
                **extra,
            )
        else:
            # zarr v2: use compressor=
            try:
                import numcodecs  # type: ignore
                compressor = numcodecs.Blosc(
                    cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
                )
            except ImportError:
                compressor = None
            self._zarr.array(
                data,
                store=self._root.store,
                path=path,
                compressor=compressor,
                overwrite=True,
                **extra,
            )

    def _write_strings(self, strings: list[str], path: str) -> None:
        """Write a string array, handling v2/v3 object codec differences."""
        arr = np.array(strings, dtype=object)
        if self._ver >= 3:
            # zarr v3: variable-length strings via dtype="str"
            try:
                self._zarr.array(
                    np.array(strings, dtype=str),
                    store=self._root.store,
                    path=path,
                    overwrite=True,
                )
            except Exception:
                # fallback: store as fixed-width numpy bytes
                self._zarr.array(
                    np.array(strings),
                    store=self._root.store,
                    path=path,
                    overwrite=True,
                )
        else:
            try:
                import numcodecs  # type: ignore
                self._zarr.array(
                    arr,
                    store=self._root.store,
                    path=path,
                    object_codec=numcodecs.VLenUTF8(),
                    overwrite=True,
                    dtype=object,
                )
            except Exception:
                self._zarr.array(
                    np.array(strings),
                    store=self._root.store,
                    path=path,
                    overwrite=True,
                )

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_subject(
        self,
        subject: str,
        X: np.ndarray,
        labels: np.ndarray,
        concept_names: list[str] | np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Write or overwrite a subject's epoch array.

        Parameters
        ----------
        subject:
            Subject identifier used as the Zarr group name.
        X:
            Float32 epoch array of shape [n_trials, n_channels, n_samples].
        labels:
            Integer concept/class labels of shape [n_trials].
        concept_names:
            Optional string labels per trial (e.g. ``"dog"``).
        meta:
            Arbitrary metadata dict stored as Zarr group attributes.
        """
        assert X.ndim == 3, f"Expected 3-D array, got {X.ndim}-D"
        assert len(labels) == X.shape[0], "labels length must match n_trials"

        grp = self._root.require_group(subject)

        # EEG array — primary data
        self._write_array(X.astype(np.float32), f"{subject}/eeg")
        logger.info(
            "Wrote %s/eeg: %s (%.1f MB uncompressed)",
            subject, X.shape, X.nbytes / 1_048_576,
        )

        # Labels
        self._write_array(labels.astype(np.int32), f"{subject}/labels")

        # Concept names — stored as JSON in attrs (avoids zarr v3 string dtype warning)
        if concept_names is not None:
            import json
            names = list(concept_names) if not isinstance(concept_names, list) else concept_names
            grp.attrs["concept_names_json"] = json.dumps(names)

        # Provenance metadata in group attrs
        if meta:
            grp.attrs.update(meta)

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_subject(
        self,
        subject: str,
        slice_: slice | None = None,
        concept_filter: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Load a subject's data from the Zarr store.

        Parameters
        ----------
        subject:
            Subject identifier.
        slice_:
            Optional trial slice (e.g. ``slice(0, 256)``).
        concept_filter:
            Optional list/set of concept IDs to load selectively without reading full dataset into memory.

        Returns
        -------
        dict with keys ``"eeg"``, ``"labels"``, ``"concept_names"`` (if present),
        and ``"meta"`` (Zarr group attrs).
        """
        if subject not in self._root:
            raise KeyError(f"Subject {subject!r} not found in {self.zarr_path}")

        grp = self._root[subject]

        if concept_filter is not None:
            all_labels = grp["labels"][:]
            mask = np.isin(all_labels, list(concept_filter))
            indices = np.where(mask)[0]
            try:
                eeg_arr = grp["eeg"].get_orthogonal_selection((indices, slice(None), slice(None)))
            except Exception:
                # Fallback in case of chunked iteration
                eeg_chunks = []
                for idx in indices:
                    eeg_chunks.append(grp["eeg"][int(idx)])
                eeg_arr = np.stack(eeg_chunks, axis=0)

            result: dict[str, Any] = {
                "eeg": eeg_arr,
                "labels": all_labels[indices],
                "meta": dict(grp.attrs),
            }
        else:
            sl = slice_ or slice(None)
            result = {
                "eeg": grp["eeg"][sl],
                "labels": grp["labels"][sl],
                "meta": dict(grp.attrs),
            }

        # concept names stored as JSON string in attrs
        if "concept_names_json" in grp.attrs:
            import json
            names = json.loads(grp.attrs["concept_names_json"])
            result["concept_names"] = np.array(names)
        elif "concept_names" in grp:
            result["concept_names"] = grp["concept_names"][:]
        return result

    def subjects(self) -> list[str]:
        """Return the list of subject group names in the store."""
        return list(self._root.keys())

    def __repr__(self) -> str:
        return f"ZarrEpochStore({self.zarr_path}, subjects={self.subjects()})"
