"""
Trial metadata schema and base dataset abstractions.

Every processed trial — regardless of dataset — carries a TrialMeta record
so that processed Zarr stores remain self-describing and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class TrialMeta:
    """Full provenance record for one EEG trial/epoch.

    All fields should be populated at epoch-creation time.  Missing optional
    fields are left as None and are excluded from Zarr attrs on serialisation.
    """

    # ── Dataset identity ────────────────────────────────────────────────────
    dataset: str                  # e.g. "ds003825", "nm000232"
    subject: str                  # e.g. "sub-01"
    session: str | None = None    # e.g. "ses-01"
    run: str | None = None        # e.g. "run-01"
    recording_id: str | None = None  # full BIDS recording identifier

    # ── Trial identity ───────────────────────────────────────────────────────
    trial_index: int | None = None   # 0-based index within the recording
    global_trial_id: str | None = None  # "{dataset}_{subject}_{run}_{trial_index}"

    # ── Stimulus / semantic identity ─────────────────────────────────────────
    stimulus_id: str | None = None   # raw events.tsv stimulus value
    concept_id: int | None = None    # THINGS concept integer id (if applicable)
    concept_name: str | None = None  # e.g. "dog"
    condition: str | None = None     # e.g. "train", "test", "imagery"

    # ── Signal parameters ────────────────────────────────────────────────────
    sampling_rate: float | None = None   # Hz after preprocessing
    n_channels: int | None = None
    tmin: float | None = None       # epoch start relative to event (s)
    tmax: float | None = None       # epoch end relative to event (s)

    # ── Quality flags ────────────────────────────────────────────────────────
    rejected: bool = False
    rejection_reason: str | None = None

    # ── Behavioural (dataset-specific) ──────────────────────────────────────
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for Zarr attrs or Parquet rows."""
        d = asdict(self)
        # Flatten extra into top level
        extra = d.pop("extra", {})
        d.update(extra)
        # Remove None values to keep attrs clean
        return {k: v for k, v in d.items() if v is not None}

    @property
    def auto_global_id(self) -> str:
        """Auto-generate a unique global trial identifier."""
        parts = [self.dataset, self.subject]
        if self.session:
            parts.append(self.session)
        if self.run:
            parts.append(self.run)
        parts.append(str(self.trial_index or 0))
        return "_".join(parts)
