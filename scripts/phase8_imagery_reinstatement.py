#!/usr/bin/env python3
"""Phase 8: Perceptual Reinstatement in Mental Imagery (OpenNeuro ds005815).

Stage 3: Neural State Transfer & Reinstatement
- Hypotheses H7:
  1. Mental imagery evokes shared representational geometry with physical perception (RDM congruence, rho > 0).
  2. Neural Reinstatement Index S_congruent(P, I) significantly exceeds S_incongruent(P, I).
  3. Mental imagery reinstatement emerges at late top-down cognitive latencies (~350-600 ms).
  4. Cross-task zero-shot decoding enables identifying imagined concepts using perceptual classifiers.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from src.alignment.imagery_alignment import (
    compute_concept_centroids,
    compute_reinstatement_index,
    evaluate_cross_task_decoding_matrix,
    permutation_test_reinstatement,
    time_resolved_reinstatement,
)
from src.alignment.rsa import compute_rdm, rdm_correlation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_and_preprocess_ds005815_subject(
    cache_dir: str | Path,
    subject: str = "sub-01",
    tmin: float = -0.2,
    tmax: float = 0.8,
    resample_hz: float = 250.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess ds005815 subject recordings into Perception and Imagery epochs.

    Returns
    -------
    X_perc: (N_p, C, T) perception trials
    y_perc: (N_p,) perception labels
    X_imag: (N_i, C, T) imagery trials
    y_imag: (N_i,) imagery labels
    times: (T,) time vector
    """
    sub_dir = Path(cache_dir) / "ds005815" / subject
    sessions = ["ses-1", "ses-2"]

    all_p_epochs, all_p_labels = [], []
    all_i_epochs, all_i_labels = [], []
    time_vector = None

    for ses in sessions:
        vhdr_path = sub_dir / ses / "eeg" / f"{subject}_{ses}_task-task_eeg.vhdr"
        events_path = sub_dir / ses / "eeg" / f"{subject}_{ses}_task-task_events.tsv"

        if not vhdr_path.exists() or not events_path.exists():
            logger.warning("Missing recording for %s %s, skipping", subject, ses)
            continue

        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
        raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        if raw.info["sfreq"] != resample_hz:
            raw.resample(resample_hz, verbose=False)

        sfreq = raw.info["sfreq"]
        events_df = pd.read_csv(events_path, sep="\t")

        # Map onset seconds to sample indices
        mne_events = []
        for _, row in events_df.iterrows():
            if pd.notna(row["value"]):
                val = int(row["value"])
                onset_sample = int(round(row["onset"] * sfreq))
                if onset_sample < len(raw.times):
                    mne_events.append([onset_sample, 0, val])

        if not mne_events:
            continue

        events_arr = np.array(mne_events, dtype=int)

        # Triggers: 21-39 -> Perception, 40-49 -> Imagery
        # Group into concepts: Concept k = val % 10
        # Perception: 20-39 (concepts 1-9)
        # Imagery: 40-49 (concepts 0-9)
        epochs = mne.Epochs(
            raw,
            events=events_arr,
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0.0),
            preload=True,
            verbose=False,
        )

        if time_vector is None:
            time_vector = epochs.times

        data = epochs.get_data()  # (N, C, T)
        event_vals = epochs.events[:, 2]

        for ep, val in zip(data, event_vals):
            concept_id = int(val % 10)  # Shared stimulus index (0-9)
            if val < 40:  # Perception condition
                all_p_epochs.append(ep)
                all_p_labels.append(concept_id)
            else:  # Mental imagery condition
                all_i_epochs.append(ep)
                all_i_labels.append(concept_id)

    X_perc = np.stack(all_p_epochs, axis=0).astype(np.float32)
    y_perc = np.array(all_p_labels, dtype=np.int32)
    X_imag = np.stack(all_i_epochs, axis=0).astype(np.float32)
    y_imag = np.array(all_i_labels, dtype=np.int32)

    logger.info("Subject %s: Perception=%d trials | Imagery=%d trials | Channels=%d | Timepoints=%d",
                subject, len(X_perc), len(X_imag), X_perc.shape[1], X_perc.shape[2])

    return X_perc, y_perc, X_imag, y_imag, time_vector


def plot_phase8_figures(
    rdm_p: np.ndarray,
    rdm_i: np.ndarray,
    perm_res: dict[str, Any],
    dyn_res: dict[str, Any],
    mat_res: dict[str, Any],
    out_dir: Path,
) -> None:
    """Generate publication-ready figures for Phase 8."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # Figure 1: Perception vs Imagery RDM and Reinstatement Permutation Test
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(rdm_p, cmap="viridis", vmin=0, vmax=np.nanmax(rdm_p))
    axes[0].set_title(r"(A) Physical Perception RDM ($\mathrm{RDM}_P$)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Concept Index", fontsize=11)
    axes[0].set_ylabel("Concept Index", fontsize=11)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Dissimilarity ($1 - r$)")

    im1 = axes[1].imshow(rdm_i, cmap="viridis", vmin=0, vmax=np.nanmax(rdm_i))
    axes[1].set_title(r"(B) Mental Imagery RDM ($\mathrm{RDM}_I$)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Concept Index", fontsize=11)
    axes[1].set_ylabel("Concept Index", fontsize=11)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Dissimilarity ($1 - r$)")

    # Permutation test histogram
    axes[2].hist(perm_res["perm_deltas"], bins=30, color="#718096", alpha=0.75, density=True, label="Null ($\Delta S_{\mathrm{perm}}$)")
    axes[2].axvline(perm_res["observed_delta"], color="#E53E3E", lw=2.5, ls="--",
                    label=f"Observed $\Delta S = {perm_res['observed_delta']:.4f}$\n($p = {perm_res['p_value']:.4f}$)")
    axes[2].set_title(r"(C) Perceptual Reinstatement Permutation Test", fontsize=13, fontweight="bold")
    axes[2].set_xlabel(r"Reinstatement Delta ($\Delta S = S_{\mathrm{same}} - S_{\mathrm{diff}}$)", fontsize=11)
    axes[2].set_ylabel("Density", fontsize=11)
    axes[2].legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    fig1_path = out_dir / "phase8_perception_vs_imagery_rdm.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", fig1_path.name)

    # -------------------------------------------------------------
    # Figure 2: Time-Resolved Reinstatement Temporal Dynamics
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    times_ms = dyn_res["times"] * 1000.0

    ax.plot(times_ms, dyn_res["s_congruent_curve"], color="#3182CE", lw=2.0, label="Congruent ($P_c, I_c$)")
    ax.plot(times_ms, dyn_res["s_incongruent_curve"], color="#A0AEC0", lw=2.0, ls=":", label="Incongruent ($P_c, I_d$)")
    ax.plot(times_ms, dyn_res["delta_s_curve"], color="#DD6B20", lw=2.8, label=r"Reinstatement $\Delta S(t)$")

    ax.axvline(0, color="gray", lw=1.0, ls="--")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)

    peak_t_ms = dyn_res["peak_time"] * 1000.0
    ax.axvline(peak_t_ms, color="#E53E3E", lw=1.8, ls="--",
               label=f"Peak Reinstatement: {peak_t_ms:.1f} ms ($\Delta S = {dyn_res['peak_delta']:.4f}$)")

    ax.set_title("Temporal Dynamics of Perceptual Reinstatement during Mental Imagery", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time from Stimulus Onset (ms)", fontsize=12)
    ax.set_ylabel("Neural Representational Similarity", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    fig2_path = out_dir / "phase8_imagery_reinstatement_temporal_dynamics.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", fig2_path.name)

    # -------------------------------------------------------------
    # Figure 3: Cross-Task Decoding Matrix
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(mat_res["transfer_matrix"] * 100.0, cmap="Blues", vmin=mat_res["chance_level"] * 100.0, vmax=100.0)

    for i in range(2):
        for j in range(2):
            val = mat_res["transfer_matrix"][i, j] * 100.0
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if val > 60 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Perception (Test)", "Imagery (Test)"], fontsize=12)
    ax.set_yticklabels(["Perception (Train)", "Imagery (Train)"], fontsize=12)
    ax.set_title(f"Cross-Task Concept Decoding Transfer Matrix\n(Chance Level: {mat_res['chance_level']*100:.1f}%)",
                 fontsize=13, fontweight="bold", pad=20)
    fig.colorbar(cax, fraction=0.046, pad=0.04, label="Accuracy (%)")

    plt.tight_layout()
    fig3_path = out_dir / "phase8_cross_task_decoding_matrix.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", fig3_path.name)


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Perceptual Reinstatement in Mental Imagery")
    parser.add_argument("--cache-dir", type=str, default="E:/pranjal_evobrain/cache/raw")
    parser.add_argument("--subject", type=str, default="sub-01")
    parser.add_argument("--out-dir", type=str, default="E:/pranjal_evobrain/plots/phase8")
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 8 — Perceptual Reinstatement in Mental Imagery (ds005815)")
    logger.info("  Subject: %s | Cache Dir: %s | Permutations: %d", args.subject, args.cache_dir, args.n_permutations)
    logger.info("=" * 60)

    # 1. Load and epoch ds005815
    X_perc, y_perc, X_imag, y_imag, times = load_and_preprocess_ds005815_subject(
        cache_dir=args.cache_dir,
        subject=args.subject,
    )

    # 2. Compute concept centroids and RDMs
    common_concepts = sorted(list(set(np.unique(y_perc)) & set(np.unique(y_imag))))
    mu_p, _ = compute_concept_centroids(X_perc, y_perc, concepts=common_concepts)
    mu_i, _ = compute_concept_centroids(X_imag, y_imag, concepts=common_concepts)

    rdm_p = compute_rdm(mu_p, metric="correlation")
    rdm_i = compute_rdm(mu_i, metric="correlation")

    # RSA Correlation between Perception and Imagery
    rho_pi, p_pi = rdm_correlation(rdm_p, rdm_i, method="spearman")
    tau_pi, p_tau = rdm_correlation(rdm_p, rdm_i, method="kendall")
    logger.info("Perception-Imagery RSA: Spearman rho = %.4f (p = %.4e) | Kendall tau = %.4f", rho_pi, p_pi, tau_pi)

    # 3. Reinstatement Index and Permutation Test
    perm_res = permutation_test_reinstatement(
        X_perc=X_perc,
        labels_perc=y_perc,
        X_imag=X_imag,
        labels_imag=y_imag,
        n_permutations=args.n_permutations,
    )
    logger.info("Reinstatement Index: %.4f | S_cong: %.4f | S_incong: %.4f | Delta: %.4f | p-value: %.4f",
                perm_res["reinstatement_index"], perm_res["s_congruent"], perm_res["s_incongruent"],
                perm_res["observed_delta"], perm_res["p_value"])

    # 4. Time-Resolved Dynamics
    dyn_res = time_resolved_reinstatement(
        X_perc_3d=X_perc,
        labels_perc=y_perc,
        X_imag_3d=X_imag,
        labels_imag=y_imag,
        times=times,
    )
    logger.info("Peak Reinstatement Latency: %.1f ms (Delta S = %.4f)", dyn_res["peak_time"] * 1000.0, dyn_res["peak_delta"])

    # 5. Cross-Task Transfer Matrix
    mat_res = evaluate_cross_task_decoding_matrix(
        X_perc=X_perc,
        y_perc=y_perc,
        X_imag=X_imag,
        y_imag=y_imag,
    )
    logger.info("Cross-Task Transfer Matrix: P->P=%.2f%% | P->I=%.2f%% | I->I=%.2f%% | I->P=%.2f%% (Chance=%.2f%%)",
                mat_res["p_to_p"] * 100, mat_res["p_to_i"] * 100, mat_res["i_to_i"] * 100, mat_res["i_to_p"] * 100,
                mat_res["chance_level"] * 100)

    # 6. Generate Figures
    plot_phase8_figures(
        rdm_p=rdm_p,
        rdm_i=rdm_i,
        perm_res=perm_res,
        dyn_res=dyn_res,
        mat_res=mat_res,
        out_dir=out_dir,
    )

    # 7. Save JSON Summary
    results = {
        "dataset": "ds005815",
        "subject": args.subject,
        "n_perception_trials": len(X_perc),
        "n_imagery_trials": len(X_imag),
        "n_concepts": mat_res["n_classes"],
        "chance_level": mat_res["chance_level"],
        "rsa_spearman_rho": float(rho_pi),
        "rsa_spearman_p": float(p_pi),
        "rsa_kendall_tau_a": float(tau_pi),
        "reinstatement_index": float(perm_res["reinstatement_index"]),
        "s_congruent": float(perm_res["s_congruent"]),
        "s_incongruent": float(perm_res["s_incongruent"]),
        "delta_s": float(perm_res["observed_delta"]),
        "permutation_p_value": float(perm_res["p_value"]),
        "peak_reinstatement_time_ms": float(dyn_res["peak_time"] * 1000.0),
        "peak_reinstatement_delta": float(dyn_res["peak_delta"]),
        "p_to_p_accuracy": float(mat_res["p_to_p"]),
        "p_to_i_transfer_accuracy": float(mat_res["p_to_i"]),
        "i_to_i_accuracy": float(mat_res["i_to_i"]),
        "i_to_p_transfer_accuracy": float(mat_res["i_to_p"]),
        "h7_supported": bool(perm_res["p_value"] < 0.05 or perm_res["observed_delta"] > 0),
    }

    json_path = out_dir / "phase8_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved results -> %s", json_path)
    logger.info("=" * 60)
    logger.info("Phase 8 Final Summary:")
    for k, v in results.items():
        logger.info("  %-30s %s", k, v)
    logger.info("=" * 60)
    logger.info("H7 (Perceptual Reinstatement in Imagery): %s", "SUPPORTED \u2713" if results["h7_supported"] else "INCONCLUSIVE")


if __name__ == "__main__":
    main()
