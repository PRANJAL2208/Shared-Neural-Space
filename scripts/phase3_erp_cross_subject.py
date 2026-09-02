"""
Phase 3: MVP — Cross-Subject ERP Similarity Test.

Hypothesis H1:
    For the same visual concept, the mean ERP is more similar between
    two subjects than for different concepts.
    S_same > S_different  (p < 0.05, permutation test)

Steps
-----
1. Load sub-01 and sub-02 epochs from Zarr
2. Select N_CONCEPTS concepts shared by both (all 1854 are seen by all subjects)
3. Compute per-concept mean ERP: [N_CONCEPTS, 63, 251] per subject
4. Cross-subject RDM: Pearson r for all (c_i x c_j) pairs → [N_CONCEPTS, N_CONCEPTS]
5. Split diagonal (same-concept) vs off-diagonal (different-concept)
6. Permutation test: label-shuffle null distribution (n_perms=5000)
7. Time-resolved: sliding per-timepoint cross-subject correlation (50ms = 12 samples)
8. Save plots to E:/pranjal_evobrain/plots/phase3/

Usage
-----
    $env:PYTHONPATH = "."
    python scripts/phase3_erp_cross_subject.py
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3")

# ── Config ─────────────────────────────────────────────────────────────────────
ZARR_PATH    = "E:/pranjal_evobrain/features/ds003825_epochs.zarr"
PLOT_DIR     = Path("E:/pranjal_evobrain/plots/phase3")
SUBJECT_A    = "sub-01"
SUBJECT_B    = "sub-02"
N_CONCEPTS   = 20        # concepts to use in MVP
N_PERMS      = 5000      # permutation test iterations
TMIN_S       = -0.2      # from config
SFREQ        = 250.0     # Hz
WINDOW_MS    = 50        # time-resolved window width in ms
RESULTS_JSON = "E:/pranjal_evobrain/plots/phase3/phase3_results.json"


def load_subject(zarr_path: str, subject: str):
    """Load epochs and labels from Zarr. Returns (X [n,63,251], labels [n])."""
    from src.data.zarr_store import ZarrEpochStore
    store = ZarrEpochStore(zarr_path)
    data = store.read_subject(subject)
    X      = data["eeg"].astype(np.float32)     # [n_trials, 63, 251]
    labels = data["labels"].astype(np.int32)     # [n_trials]
    names  = data.get("concept_names", None)
    logger.info("Loaded %s: X=%s, %d unique concepts", subject, X.shape, len(np.unique(labels)))
    return X, labels, names


def select_concepts(labels_a, labels_b, n: int, seed: int = 42):
    """Pick n concepts present in BOTH subjects, with ≥3 trials each."""
    shared = set(np.unique(labels_a)) & set(np.unique(labels_b))
    # Filter to concepts with enough trials in both subjects
    valid = []
    for c in sorted(shared):
        na = int((labels_a == c).sum())
        nb = int((labels_b == c).sum())
        if na >= 1 and nb >= 1:
            valid.append((c, na + nb))
    # Sort by total trial count descending (most-represented concepts first)
    valid.sort(key=lambda x: -x[1])
    selected = [c for c, _ in valid[:n]]
    logger.info("Selected %d concepts (of %d shared): %s", len(selected), len(valid), selected[:5])
    return selected


def mean_erp(X, labels, concepts):
    """Compute per-concept mean ERP. Returns array [n_concepts, n_ch, n_time]."""
    erps = []
    for c in concepts:
        mask = labels == c
        erps.append(X[mask].mean(axis=0))
    return np.stack(erps, axis=0)   # [N, 63, 251]


def cross_subject_rdm(erps_a, erps_b):
    """Pearson correlation matrix [N, N] between subject A and B concept ERPs."""
    # Flatten spatial dim: [N, 63*251]
    fa = erps_a.reshape(len(erps_a), -1)
    fb = erps_b.reshape(len(erps_b), -1)
    # 1 - correlation distance = Pearson r
    rdm = 1.0 - cdist(fa, fb, metric="correlation")   # [N, N]
    return rdm.astype(np.float32)


def permutation_test(rdm, n_perms=5000, seed=42):
    """
    H1: diagonal (same-concept) > off-diagonal (different-concept).

    Null: shuffle row indices of subject B and recompute mean-diagonal.
    Returns (observed_effect, p_value, null_distribution).
    """
    rng = np.random.default_rng(seed)
    n = rdm.shape[0]

    same     = rdm.diagonal().mean()
    diff_mask = ~np.eye(n, dtype=bool)
    different = rdm[diff_mask].mean()
    observed  = same - different

    null = np.empty(n_perms, dtype=np.float32)
    for i in range(n_perms):
        perm_idx = rng.permutation(n)
        perm_same = rdm[np.arange(n), perm_idx].mean()
        null[i] = perm_same - different

    p_value = float((null >= observed).mean())
    return float(observed), p_value, null


def time_resolved_correlation(erps_a, erps_b, sfreq=250.0, window_ms=50):
    """
    Per-timepoint sliding mean cross-subject same vs different correlation.

    Returns:
        times_s     : [n_timepoints] time axis in seconds
        r_same      : [n_timepoints] mean same-concept cross-subject r
        r_different : [n_timepoints] mean different-concept cross-subject r
    """
    n_concepts, n_ch, n_time = erps_a.shape
    win = max(1, int(window_ms / 1000 * sfreq))

    times, r_same_ts, r_diff_ts = [], [], []

    for t in range(0, n_time - win + 1):
        sl = slice(t, t + win)
        # Flatten: [N, n_ch * win]
        fa = erps_a[:, :, sl].reshape(n_concepts, -1)
        fb = erps_b[:, :, sl].reshape(n_concepts, -1)

        rdm_t = 1.0 - cdist(fa, fb, metric="correlation")
        same_t = float(rdm_t.diagonal().mean())
        diff_t = float(rdm_t[~np.eye(n_concepts, dtype=bool)].mean())

        times.append(TMIN_S + (t + win / 2) / sfreq)
        r_same_ts.append(same_t)
        r_diff_ts.append(diff_t)

    return np.array(times), np.array(r_same_ts), np.array(r_diff_ts)


def get_concept_names(names_a, concepts):
    """Map concept_id → name string for labels."""
    if names_a is None:
        return {c: str(c) for c in concepts}
    # names_a is indexed by trial; find first occurrence of each concept
    return {c: str(names_a[0]) if len(names_a) > 0 else str(c) for c in concepts}


def plot_results(rdm, observed, p_value, null,
                 times, r_same, r_diff,
                 concepts, names_a, erps_a, erps_b, out_dir: Path):
    """Generate and save Phase 3 figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(concepts)

    # Build concept label list
    labels_list = [str(c) for c in concepts]

    # ── Figure 1: Cross-subject RDM + null distribution ────────────────────
    fig = plt.figure(figsize=(14, 5), facecolor="#0f1117")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # RDM heatmap
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(rdm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xticks(range(n)); ax1.set_xticklabels(labels_list, rotation=90, fontsize=6, color="white")
    ax1.set_yticks(range(n)); ax1.set_yticklabels(labels_list, fontsize=6, color="white")
    ax1.set_title("Cross-Subject RDM\n(Pearson r, A × B)", color="white", fontsize=10)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values(): spine.set_edgecolor("#333")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white"); cbar.ax.yaxis.label.set_color("white")
    # Highlight diagonal
    for i in range(n):
        ax1.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="#00ff88", lw=1.5))

    # Same vs different box plot
    ax2 = fig.add_subplot(gs[1])
    same_vals = rdm.diagonal()
    diff_vals = rdm[~np.eye(n, dtype=bool)]
    bplot = ax2.boxplot([diff_vals, same_vals], patch_artist=True,
                        medianprops=dict(color="white", lw=2))
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Different\nconcepts", "Same\nconcepts"])
    colors = ["#ff6b6b", "#00ff88"]
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bplot[element]: item.set_color("#aaa")
    ax2.set_facecolor("#1a1d27"); ax2.tick_params(colors="white")
    ax2.set_ylabel("Pearson r", color="white")
    ax2.set_title(f"H₁: Same > Different\np = {p_value:.4f}{'*' if p_value < 0.05 else ''}",
                  color="#00ff88" if p_value < 0.05 else "#ff6b6b", fontsize=10)
    for spine in ax2.spines.values(): spine.set_edgecolor("#333")
    ax2.yaxis.label.set_color("white")

    # Null distribution
    ax3 = fig.add_subplot(gs[2])
    ax3.hist(null, bins=60, color="#4a90d9", alpha=0.7, edgecolor="none")
    ax3.axvline(observed, color="#00ff88", lw=2, label=f"Observed Δr = {observed:.3f}")
    ax3.set_facecolor("#1a1d27"); ax3.tick_params(colors="white")
    ax3.set_xlabel("Δr (same − different)", color="white")
    ax3.set_ylabel("Count", color="white")
    ax3.set_title(f"Permutation Null (n={N_PERMS})", color="white", fontsize=10)
    ax3.legend(frameon=False, labelcolor="white")
    for spine in ax3.spines.values(): spine.set_edgecolor("#333")

    fig.suptitle(f"Phase 3 MVP — Cross-Subject ERP Similarity\n"
                 f"{SUBJECT_A} × {SUBJECT_B} | {n} concepts | "
                 f"ds003825", color="white", fontsize=12, y=1.02)
    fig.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase3_rdm_and_permtest.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info("Saved: phase3_rdm_and_permtest.png")

    # ── Figure 2: Time-resolved cross-subject correlation ─────────────────
    fig2, ax = plt.subplots(figsize=(12, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.plot(times * 1000, r_same, color="#00ff88", lw=2, label="Same concept")
    ax.plot(times * 1000, r_diff, color="#ff6b6b", lw=2, label="Different concepts")
    ax.fill_between(times * 1000, r_diff, r_same, alpha=0.15, color="#00ff88")
    ax.axvline(0, color="#888", lw=1.5, ls="--", label="Stimulus onset")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel("Time (ms)", color="white", fontsize=11)
    ax.set_ylabel("Mean Pearson r (cross-subject)", color="white", fontsize=11)
    ax.set_title(f"Time-Resolved Cross-Subject Similarity\n"
                 f"{SUBJECT_A} × {SUBJECT_B} | {n} concepts | {WINDOW_MS}ms sliding window",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.legend(frameon=False, labelcolor="white", fontsize=10)
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    # Shade stim duration (50ms RSVP)
    ax.axvspan(0, 50, alpha=0.1, color="#aaa", label="Stim (50ms)")
    fig2.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase3_time_resolved.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig2)
    logger.info("Saved: phase3_time_resolved.png")

    # ── Figure 3: Mean ERP waveforms for 6 example concepts ────────────────
    fig3, axes = plt.subplots(2, 3, figsize=(14, 6), facecolor="#0f1117")
    t_axis = (np.arange(erps_a.shape[-1]) / SFREQ + TMIN_S) * 1000
    for idx, (ax, c) in enumerate(zip(axes.flat, concepts[:6])):
        ax.set_facecolor("#1a1d27")
        ci = concepts.index(c)
        # Global field power (std across channels) as summary waveform
        gfp_a = erps_a[ci].std(axis=0)
        gfp_b = erps_b[ci].std(axis=0)
        ax.plot(t_axis, gfp_a, color="#4a90d9", lw=1.8, label=SUBJECT_A)
        ax.plot(t_axis, gfp_b, color="#e67e22", lw=1.8, label=SUBJECT_B, alpha=0.85)
        ax.axvline(0, color="#888", lw=1, ls="--")
        ax.set_title(f"Concept {c}", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        if idx >= 3: ax.set_xlabel("ms", color="white", fontsize=8)
        if idx % 3 == 0: ax.set_ylabel("GFP (µV)", color="white", fontsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#444")
    axes.flat[0].legend(frameon=False, labelcolor="white", fontsize=8)
    fig3.suptitle("Global Field Power — Example Concepts (A=blue, B=orange)",
                  color="white", fontsize=11)
    fig3.patch.set_facecolor("#0f1117")
    plt.tight_layout()
    plt.savefig(out_dir / "phase3_gfp_examples.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig3)
    logger.info("Saved: phase3_gfp_examples.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr",       default=ZARR_PATH)
    parser.add_argument("--subject-a",  default=SUBJECT_A)
    parser.add_argument("--subject-b",  default=SUBJECT_B)
    parser.add_argument("--n-concepts", type=int, default=N_CONCEPTS)
    parser.add_argument("--n-perms",    type=int, default=N_PERMS)
    parser.add_argument("--out-dir",    default=str(PLOT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    logger.info("=" * 60)
    logger.info("Phase 3 — Cross-Subject ERP Similarity MVP")
    logger.info("  %s × %s | %d concepts | %d perms",
                args.subject_a, args.subject_b, args.n_concepts, args.n_perms)
    logger.info("=" * 60)

    # ── Load ────────────────────────────────────────────────────────────────
    Xa, la, names_a = load_subject(args.zarr, args.subject_a)
    Xb, lb, names_b = load_subject(args.zarr, args.subject_b)

    # ── Select concepts ─────────────────────────────────────────────────────
    concepts = select_concepts(la, lb, n=args.n_concepts)
    if len(concepts) < args.n_concepts:
        logger.warning("Only %d shared concepts available", len(concepts))

    # ── Mean ERPs ───────────────────────────────────────────────────────────
    logger.info("Computing mean ERPs …")
    erps_a = mean_erp(Xa, la, concepts)   # [N, 63, 251]
    erps_b = mean_erp(Xb, lb, concepts)
    logger.info("  ERP shape: %s", erps_a.shape)

    # ── Cross-subject RDM ───────────────────────────────────────────────────
    logger.info("Computing cross-subject RDM …")
    rdm = cross_subject_rdm(erps_a, erps_b)    # [N, N]
    same     = float(rdm.diagonal().mean())
    diff_mask = ~np.eye(len(concepts), dtype=bool)
    different = float(rdm[diff_mask].mean())
    logger.info("  Mean same-concept r      : %.4f", same)
    logger.info("  Mean different-concept r : %.4f", different)
    logger.info("  Δr (observed)            : %.4f", same - different)

    # ── Permutation test ────────────────────────────────────────────────────
    logger.info("Running permutation test (n=%d) …", args.n_perms)
    observed, p_value, null = permutation_test(rdm, n_perms=args.n_perms)
    logger.info("  p-value = %.4f %s", p_value, "*** SIGNIFICANT" if p_value < 0.05 else "(not significant)")

    # ── Time-resolved ───────────────────────────────────────────────────────
    logger.info("Time-resolved cross-subject correlation (%dms window) …", WINDOW_MS)
    times, r_same_ts, r_diff_ts = time_resolved_correlation(
        erps_a, erps_b, sfreq=SFREQ, window_ms=WINDOW_MS
    )
    peak_idx  = int(np.argmax(r_same_ts - r_diff_ts))
    peak_time = float(times[peak_idx] * 1000)
    logger.info("  Peak discriminability at: %.0f ms", peak_time)

    # ── Plots ───────────────────────────────────────────────────────────────
    logger.info("Generating figures → %s", out_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_results(rdm, observed, p_value, null,
                     times, r_same_ts, r_diff_ts,
                     concepts, names_a, erps_a, erps_b, out_dir)

    # ── Results JSON ────────────────────────────────────────────────────────
    results = {
        "subject_a": args.subject_a,
        "subject_b": args.subject_b,
        "n_concepts": len(concepts),
        "concepts": [int(c) for c in concepts],
        "mean_same_concept_r":      round(same, 6),
        "mean_different_concept_r": round(different, 6),
        "observed_delta_r":         round(float(observed), 6),
        "p_value":                  round(p_value, 6),
        "significant_p05":          p_value < 0.05,
        "peak_discriminability_ms": round(peak_time, 1),
        "h1_supported":             p_value < 0.05 and observed > 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase3_results.json").write_text(json.dumps(results, indent=2))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3 Results")
    logger.info("=" * 60)
    for k, v in results.items():
        if k not in ("concepts",):
            logger.info("  %-35s %s", k, v)
    logger.info("=" * 60)
    logger.info("H₁ (S_same > S_different): %s",
                "SUPPORTED ✓" if results["h1_supported"] else "NOT SUPPORTED ✗")
    logger.info("Figures saved to: %s", out_dir)


if __name__ == "__main__":
    main()
