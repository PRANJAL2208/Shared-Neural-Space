"""
Phase 4: Representational Similarity Analysis (RSA) & Geometric Manifold Alignment.

Hypothesis H2 (Stage 1 RSA):
    The pairwise representational geometry of concepts is preserved across
    different human brains:
    ρ(RDM_A, RDM_B) > 0  (p < 0.05, label-permutation test)

Steps
-----
1. Load preprocessed epochs for sub-01 and sub-02 from Zarr.
2. Select shared concepts (default N=50).
3. Compute within-subject mean ERPs: [N_CONCEPTS, 63, 251].
4. Compute within-subject RDMs: RDM_A, RDM_B [N_CONCEPTS, N_CONCEPTS].
5. Evaluate Second-Order Geometry Alignment:
   - Spearman rank correlation ρ
   - Kendall's τ_a
   - Pearson correlation r
6. Non-parametric condition permutation test (N=5000 iterations).
7. Time-resolved second-order RSA trajectory across sliding windows.
8. Multidimensional Scaling (MDS) & Orthogonal Procrustes manifold alignment.
9. Generate publication-grade dark-mode visualizations and structured JSON output.

Usage
-----
    $env:PYTHONPATH = "."
    python scripts/phase4_rsa_geometry.py --n-concepts 50 --n-perms 5000
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
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist

from src.data.zarr_store import ZarrEpochStore
from src.alignment.rsa import (
    compute_rdm,
    vectorize_rdm,
    rdm_correlation,
    kendall_tau_a,
    permutation_rdm_test,
    time_resolved_rsa,
    compute_mds_embeddings,
    procrustes_alignment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4")

ZARR_PATH    = "E:/pranjal_evobrain/features/ds003825_epochs.zarr"
PLOT_DIR     = Path("E:/pranjal_evobrain/plots/phase4")
SUBJECT_A    = "sub-01"
SUBJECT_B    = "sub-02"
N_CONCEPTS   = 50
N_PERMS      = 5000
TMIN_S       = -0.2
SFREQ        = 250.0
WINDOW_MS    = 50


def load_subject(zarr_path: str, subject: str):
    """Load epochs and labels from Zarr store."""
    store = ZarrEpochStore(zarr_path)
    data = store.read_subject(subject)
    X      = data["eeg"].astype(np.float32)
    labels = data["labels"].astype(np.int32)
    names  = data.get("concept_names", None)
    logger.info("Loaded %s: X=%s, %d unique concepts", subject, X.shape, len(np.unique(labels)))
    return X, labels, names


def select_concepts(labels_a, labels_b, n: int):
    """Select top n concepts with highest trial representations in both subjects."""
    shared = set(np.unique(labels_a)) & set(np.unique(labels_b))
    valid = []
    for c in sorted(shared):
        na = int((labels_a == c).sum())
        nb = int((labels_b == c).sum())
        if na >= 1 and nb >= 1:
            valid.append((c, na + nb))
    valid.sort(key=lambda x: -x[1])
    selected = [c for c, _ in valid[:n]]
    logger.info("Selected %d concepts (of %d shared): %s", len(selected), len(valid), selected[:5])
    return selected


def compute_mean_erps(X, labels, concepts):
    """Compute concept-wise mean ERP array [N_CONCEPTS, n_channels, n_samples]."""
    erps = []
    for c in concepts:
        mask = (labels == c)
        erps.append(X[mask].mean(axis=0))
    return np.stack(erps, axis=0).astype(np.float32)


def plot_phase4_figures(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    spearman_rho: float,
    kendall_tau: float,
    pearson_r: float,
    perm_result: dict,
    times: np.ndarray,
    rhos_time: np.ndarray,
    pvals_time: np.ndarray,
    emb_a: np.ndarray,
    emb_b_aligned: np.ndarray,
    disparity: float,
    concepts: list[int],
    out_dir: Path,
):
    """Generate dark-mode publication figures for Phase 4."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(concepts)
    vec_a = vectorize_rdm(rdm_a)
    vec_b = vectorize_rdm(rdm_b)
    labels_str = [f"c{c}" for c in concepts]

    # ── Figure 1: Second-Order RDM Alignment & Permutation Null ──────────────
    fig1 = plt.figure(figsize=(16, 5), facecolor="#0f1117")
    gs1 = gridspec.GridSpec(1, 4, figure=fig1, width_ratios=[1, 1, 1.2, 1], wspace=0.35)

    # RDM A
    ax1 = fig1.add_subplot(gs1[0])
    im1 = ax1.imshow(rdm_a, cmap="viridis", aspect="auto")
    ax1.set_title(f"Subject A ({SUBJECT_A}) RDM\n[1 - Pearson r]", color="white", fontsize=10)
    ax1.set_xlabel("Concepts", color="white", fontsize=8)
    ax1.set_ylabel("Concepts", color="white", fontsize=8)
    ax1.tick_params(colors="white", labelsize=7)
    for spine in ax1.spines.values(): spine.set_edgecolor("#333")
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).ax.tick_params(colors="white")

    # RDM B
    ax2 = fig1.add_subplot(gs1[1])
    im2 = ax2.imshow(rdm_b, cmap="viridis", aspect="auto")
    ax2.set_title(f"Subject B ({SUBJECT_B}) RDM\n[1 - Pearson r]", color="white", fontsize=10)
    ax2.set_xlabel("Concepts", color="white", fontsize=8)
    ax2.tick_params(colors="white", labelsize=7)
    for spine in ax2.spines.values(): spine.set_edgecolor("#333")
    fig1.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors="white")

    # Vector Scatter & Fit
    ax3 = fig1.add_subplot(gs1[2])
    ax3.set_facecolor("#1a1d27")
    ax3.scatter(vec_a, vec_b, alpha=0.35, color="#00ff88", s=12, edgecolors="none")
    # Linear fit line
    m, b = np.polyfit(vec_a, vec_b, 1)
    xs = np.linspace(vec_a.min(), vec_a.max(), 100)
    ax3.plot(xs, m * xs + b, color="#ffffff", lw=1.8, label=f"Fit (Slope: {m:.2f})")
    ax3.set_xlabel("RDM Distance (Subject A)", color="white", fontsize=9)
    ax3.set_ylabel("RDM Distance (Subject B)", color="white", fontsize=9)
    ax3.set_title(f"Geometry Correlation\nSpearman ρ = {spearman_rho:.4f} (p={perm_result['p_value']:.4f})",
                  color="#00ff88" if perm_result["p_value"] < 0.05 else "#ff6b6b", fontsize=10)
    ax3.tick_params(colors="white", labelsize=8)
    ax3.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax3.spines.values(): spine.set_edgecolor("#333")

    # Permutation Null Distribution
    ax4 = fig1.add_subplot(gs1[3])
    ax4.set_facecolor("#1a1d27")
    null_dist = perm_result["null_distribution"]
    ax4.hist(null_dist, bins=50, color="#4a90d9", alpha=0.7, edgecolor="none")
    ax4.axvline(spearman_rho, color="#00ff88", lw=2, label=f"Observed ρ = {spearman_rho:.4f}")
    ax4.set_xlabel("Null Spearman ρ", color="white", fontsize=9)
    ax4.set_ylabel("Permutations", color="white", fontsize=9)
    ax4.set_title(f"Permutation Null (N={len(null_dist)})\np = {perm_result['p_value']:.4f}",
                  color="white", fontsize=10)
    ax4.tick_params(colors="white", labelsize=8)
    ax4.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax4.spines.values(): spine.set_edgecolor("#333")

    fig1.suptitle(f"Phase 4 — Second-Order Representational Similarity Analysis (RSA)\n"
                  f"{SUBJECT_A} × {SUBJECT_B} | {n} concepts | OpenNeuro ds003825",
                  color="white", fontsize=12, y=1.03)
    fig1.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase4_rdm_second_order_alignment.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig1)
    logger.info("Saved: phase4_rdm_second_order_alignment.png")

    # ── Figure 2: Time-Resolved RSA Trajectory ───────────────────────────────
    fig2, ax_t = plt.subplots(figsize=(12, 5), facecolor="#0f1117")
    ax_t.set_facecolor("#1a1d27")
    times_ms = times * 1000.0

    ax_t.plot(times_ms, rhos_time, color="#00ff88", lw=2.2, label="Second-Order RSA (Spearman ρ)")
    ax_t.fill_between(times_ms, 0, rhos_time, where=(rhos_time > 0), color="#00ff88", alpha=0.15)
    ax_t.axvline(0, color="#888888", lw=1.5, ls="--", label="Stimulus Onset")
    ax_t.axhline(0, color="#555555", lw=0.8)
    ax_t.axvspan(0, 50, alpha=0.1, color="#aaaaaa", label="RSVP Stimulus (50ms)")

    # Highlight peak latency
    peak_idx = int(np.argmax(rhos_time))
    peak_t = times_ms[peak_idx]
    peak_rho = rhos_time[peak_idx]
    ax_t.scatter([peak_t], [peak_rho], color="#ff6b6b", s=60, zorder=5)
    ax_t.annotate(f"Peak ρ = {peak_rho:.4f}\n@ {peak_t:.0f} ms",
                  xy=(peak_t, peak_rho), xytext=(peak_t + 25, peak_rho * 0.85),
                  color="#ffffff", fontsize=9,
                  arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.2))

    ax_t.set_xlabel("Time (ms)", color="white", fontsize=11)
    ax_t.set_ylabel("RDM Alignment (Spearman ρ)", color="white", fontsize=11)
    ax_t.set_title(f"Time-Resolved Representational Geometry Alignment\n"
                   f"{SUBJECT_A} × {SUBJECT_B} | {n} concepts | {WINDOW_MS}ms sliding window",
                   color="white", fontsize=12)
    ax_t.tick_params(colors="white")
    ax_t.legend(frameon=False, labelcolor="white", fontsize=9)
    for spine in ax_t.spines.values(): spine.set_edgecolor("#444")
    fig2.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase4_time_resolved_rsa_geometry.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig2)
    logger.info("Saved: phase4_time_resolved_rsa_geometry.png")

    # ── Figure 3: MDS Concept Manifold & Procrustes Geometry ─────────────────
    fig3, (ax_m1, ax_m2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f1117")

    # Procrustes Aligned 2D MDS Space
    ax_m1.set_facecolor("#1a1d27")
    ax_m1.scatter(emb_a[:, 0], emb_a[:, 1], color="#4a90d9", s=45, label=f"Subject A ({SUBJECT_A})", alpha=0.85)
    ax_m1.scatter(emb_b_aligned[:, 0], emb_b_aligned[:, 1], color="#e67e22", s=45, label=f"Subject B ({SUBJECT_B}, Aligned)", alpha=0.85)

    # Draw connection lines between corresponding concepts
    for i in range(min(n, 30)):
        ax_m1.plot([emb_a[i, 0], emb_b_aligned[i, 0]],
                   [emb_a[i, 1], emb_b_aligned[i, 1]],
                   color="#888888", lw=0.8, alpha=0.5)

    ax_m1.set_title(f"Orthogonal Procrustes Aligned Manifold\nDisparity Metric M² = {disparity:.4f}",
                    color="white", fontsize=10)
    ax_m1.set_xlabel("MDS Dimension 1", color="white", fontsize=9)
    ax_m1.set_ylabel("MDS Dimension 2", color="white", fontsize=9)
    ax_m1.tick_params(colors="white", labelsize=8)
    ax_m1.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax_m1.spines.values(): spine.set_edgecolor("#333")

    # Pairwise Distance Scatter
    ax_m2.set_facecolor("#1a1d27")
    dist_a_p = pdist(emb_a)
    dist_b_p = pdist(emb_b_aligned)
    ax_m2.scatter(dist_a_p, dist_b_p, color="#9b59b6", s=15, alpha=0.4, edgecolors="none")
    m_p, b_p = np.polyfit(dist_a_p, dist_b_p, 1)
    xs_p = np.linspace(dist_a_p.min(), dist_a_p.max(), 100)
    ax_m2.plot(xs_p, m_p * xs_p + b_p, color="#ffffff", lw=1.5, label=f"Manifold Congruence (r={pearsonr(dist_a_p, dist_b_p)[0]:.3f})")
    ax_m2.set_title("Pairwise Metric Distance Congruence in MDS Space", color="white", fontsize=10)
    ax_m2.set_xlabel("MDS Distance (Subject A)", color="white", fontsize=9)
    ax_m2.set_ylabel("MDS Distance (Subject B)", color="white", fontsize=9)
    ax_m2.tick_params(colors="white", labelsize=8)
    ax_m2.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax_m2.spines.values(): spine.set_edgecolor("#333")

    fig3.suptitle(f"Phase 4 — 2D Metric Manifold Geometry & Procrustes Interoperability\n"
                  f"{SUBJECT_A} × {SUBJECT_B} | {n} concepts", color="white", fontsize=12, y=1.02)
    fig3.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase4_mds_procrustes_manifold.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig3)
    logger.info("Saved: phase4_mds_procrustes_manifold.png")


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
    logger.info("Phase 4 — Representational Similarity Analysis & Geometry Alignment")
    logger.info("  %s × %s | %d concepts | %d perms",
                args.subject_a, args.subject_b, args.n_concepts, args.n_perms)
    logger.info("=" * 60)

    # 1. Load subjects
    Xa, la, names_a = load_subject(args.zarr, args.subject_a)
    Xb, lb, names_b = load_subject(args.zarr, args.subject_b)

    # 2. Select concepts
    concepts = select_concepts(la, lb, n=args.n_concepts)

    # 3. Compute Mean ERPs
    logger.info("Computing concept-wise mean ERPs …")
    erps_a = compute_mean_erps(Xa, la, concepts)  # [N, 63, 251]
    erps_b = compute_mean_erps(Xb, lb, concepts)

    # 4. Compute within-subject RDMs
    logger.info("Computing within-subject RDMs (1 - correlation) …")
    rdm_a = compute_rdm(erps_a, metric="correlation")
    rdm_b = compute_rdm(erps_b, metric="correlation")

    # 5. Evaluate Second-Order Geometry Alignment
    rho, p_spearman = rdm_correlation(rdm_a, rdm_b, method="spearman")
    r_val, p_pearson = rdm_correlation(rdm_a, rdm_b, method="pearson")
    tau, p_kendall   = rdm_correlation(rdm_a, rdm_b, method="kendall")

    logger.info("Second-Order RDM Alignment Results:")
    logger.info("  Spearman ρ   : %.4f (p = %.4e)", rho, p_spearman)
    logger.info("  Kendall's τ_a: %.4f (p = %.4e)", tau, p_kendall)
    logger.info("  Pearson r    : %.4f (p = %.4e)", r_val, p_pearson)

    # 6. Non-parametric condition permutation test (H2)
    logger.info("Running condition permutation test (N=%d) …", args.n_perms)
    perm_result = permutation_rdm_test(rdm_a, rdm_b, n_permutations=args.n_perms, method="spearman")
    logger.info("  Permutation p-value: %.4f %s",
                perm_result["p_value"],
                "*** STATISTICALLY SIGNIFICANT" if perm_result["p_value"] < 0.05 else "(Not Significant)")

    # 7. Time-resolved Second-Order RSA
    logger.info("Computing time-resolved second-order RSA trajectory (%dms window) …", WINDOW_MS)
    times, rhos_time, pvals_time = time_resolved_rsa(
        erps_a, erps_b, sfreq=SFREQ, window_ms=WINDOW_MS, tmin_s=TMIN_S, method="spearman"
    )
    peak_idx = int(np.argmax(rhos_time))
    peak_time_ms = float(times[peak_idx] * 1000.0)
    peak_rho_val = float(rhos_time[peak_idx])
    logger.info("  Peak RSA alignment: ρ = %.4f at %.0f ms post-stimulus", peak_rho_val, peak_time_ms)

    # 7b. Core Perceptual/Semantic Window RSA (150ms - 350ms)
    t_start_idx = int((0.15 - TMIN_S) * SFREQ)
    t_end_idx   = int((0.35 - TMIN_S) * SFREQ)
    erps_a_peak = erps_a[:, :, t_start_idx:t_end_idx]
    erps_b_peak = erps_b[:, :, t_start_idx:t_end_idx]
    rdm_a_peak = compute_rdm(erps_a_peak, metric="correlation")
    rdm_b_peak = compute_rdm(erps_b_peak, metric="correlation")
    rho_peak_win, p_peak_win = rdm_correlation(rdm_a_peak, rdm_b_peak, method="spearman")
    perm_peak_win = permutation_rdm_test(rdm_a_peak, rdm_b_peak, n_permutations=args.n_perms, method="spearman")
    logger.info("Core Visual Window (150–350 ms) RSA:")
    logger.info("  Spearman ρ: %.4f (p = %.4f, perm p = %.4f) %s",
                rho_peak_win, p_peak_win, perm_peak_win["p_value"],
                "*** SIGNIFICANT" if perm_peak_win["p_value"] < 0.05 else "")

    # 8. MDS & Procrustes Geometry (using core perceptual window)
    logger.info("Computing 2D MDS manifold embeddings and Procrustes alignment …")
    emb_a = compute_mds_embeddings(rdm_a_peak, n_components=2)
    emb_b = compute_mds_embeddings(rdm_b_peak, n_components=2)
    emb_a_norm, emb_b_aligned, disparity = procrustes_alignment(emb_a, emb_b)
    logger.info("  Procrustes disparity M²: %.4f", disparity)

    # 9. Plot Figures
    logger.info("Generating publication figures → %s", out_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_phase4_figures(
            rdm_a_peak, rdm_b_peak, rho_peak_win, tau, r_val, perm_peak_win,
            times, rhos_time, pvals_time,
            emb_a_norm, emb_b_aligned, disparity,
            concepts, out_dir
        )

    # 10. Save JSON output
    results = {
        "subject_a": args.subject_a,
        "subject_b": args.subject_b,
        "n_concepts": len(concepts),
        "concepts": [int(c) for c in concepts],
        "full_epoch_spearman_rho": round(rho, 6),
        "full_epoch_permutation_p": round(perm_result["p_value"], 6),
        "peak_window_150_350ms_spearman_rho": round(rho_peak_win, 6),
        "peak_window_permutation_p": round(perm_peak_win["p_value"], 6),
        "peak_window_significant_p05": perm_peak_win["p_value"] < 0.05,
        "peak_time_resolved_latency_ms": round(peak_time_ms, 1),
        "peak_time_resolved_rho": round(peak_rho_val, 6),
        "procrustes_disparity_m2": round(disparity, 6),
        "h2_supported": perm_peak_win["p_value"] < 0.05 and rho_peak_win > 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase4_results.json").write_text(json.dumps(results, indent=2))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 4 Final Results Summary")
    logger.info("=" * 60)
    for k, v in results.items():
        if k != "concepts":
            logger.info("  %-32s %s", k, v)
    logger.info("=" * 60)
    logger.info("H₂ (Geometry Preservation): %s",
                "SUPPORTED ✓" if results["h2_supported"] else "NOT SUPPORTED ✗")
    logger.info("All artifacts generated at: %s", out_dir)


if __name__ == "__main__":
    main()
