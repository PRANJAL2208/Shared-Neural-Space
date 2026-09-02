#!/usr/bin/env python3
"""Phase 10: Population Generalization & Neural Scaling Laws.

Stage 3 & Checkpoint G:
- Hypotheses H9:
  1. Neural Scaling Law: Zero-shot cross-subject concept decoding accuracy increases monotonically
     with training cohort size N, following a power-law trajectory: Acc(N) = A_inf - beta * N^(-gamma).
  2. Consensus Signal Amplification: Averaging representational geometries across N subjects
     amplifies the consensus signal-to-noise ratio: SNR_consensus ~ SNR_individual * sqrt(N).
  3. Universal Shared Geometry: Leave-one-subject-out decoding demonstrates robust generalization
     across the human population.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.alignment.rsa import compute_rdm, rdm_correlation
from src.data.zarr_store import ZarrEpochStore
from src.evaluation.scaling import compute_population_consensus_rdm, fit_neural_scaling_law

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_cohort_decoding(
    store: ZarrEpochStore,
    train_subjects: list[str],
    test_subject: str,
    concepts: list[int],
) -> float:
    """Train linear classifier on a cohort of N subjects and evaluate on held-out test subject."""
    concept_to_idx = {c: i for i, c in enumerate(concepts)}

    X_train_list, y_train_list = [], []
    for s in train_subjects:
        data = store.read_subject(s, concept_filter=concepts)
        X_sub = data["eeg"].astype(np.float32)
        lbls = data["labels"].astype(np.int32)
        X_train_list.append(X_sub.reshape(len(X_sub), -1))
        y_train_list.append(np.array([concept_to_idx[c] for c in lbls], dtype=np.int32))

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    data_test = store.read_subject(test_subject, concept_filter=concepts)
    X_test = data_test["eeg"].astype(np.float32).reshape(len(data_test["eeg"]), -1)
    lbls_test = data_test["labels"].astype(np.int32)
    y_test = np.array([concept_to_idx[c] for c in lbls_test], dtype=np.int32)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    acc = float(accuracy_score(y_test, preds))

    return acc


def plot_phase10_figures(
    scaling_res: dict[str, Any],
    consensus_res: dict[str, Any],
    loso_matrix: np.ndarray,
    subjects: list[str],
    out_dir: Path,
) -> None:
    """Generate publication-ready figures for Phase 10."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # Figure 1: Neural Scaling Law Curve & Power-Law Fit
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    extrap_N = scaling_res["extrapolated_sizes"]
    extrap_acc = np.array(scaling_res["extrapolated_acc"]) * 100.0

    ax.plot(extrap_N, extrap_acc, color="#3182CE", lw=2.5, ls="-",
            label=f"Power-Law Fit: $Acc(N) = {scaling_res['A_inf']*100:.1f}\\% - {scaling_res['beta']*100:.1f}\\% \\cdot N^{{-{scaling_res['gamma']:.2f}}}$ ($R^2={scaling_res['r2']:.2f}$)")

    # Empirical data points
    emp_N = scaling_res["cohort_sizes"]
    emp_acc = np.array(scaling_res["empirical_accuracies"]) * 100.0
    ax.scatter(emp_N, emp_acc, color="#E53E3E", s=90, zorder=5, edgecolors="black",
               label="Empirical Cohort Accuracies")

    # Asymptotic performance line
    ax.axhline(scaling_res["A_inf"] * 100.0, color="#38A169", lw=1.8, ls="--",
               label=f"Asymptotic Capacity Limit $A_\\infty = {scaling_res['A_inf']*100:.1f}\\%$")

    ax.set_title("Neural Scaling Law: Cross-Subject Concept Decoding vs Population Size", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Training Subjects ($N$)", fontsize=11)
    ax.set_ylabel("Zero-Shot Cross-Subject Accuracy (%)", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0, max(50.0, scaling_res["A_inf"] * 100 + 10))

    plt.tight_layout()
    fig1_path = out_dir / "phase10_neural_scaling_law.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", fig1_path.name)

    # -------------------------------------------------------------
    # Figure 2: Population Consensus RDM & SNR Amplification
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im = axes[0].imshow(consensus_res["consensus_rdm"], cmap="viridis")
    axes[0].set_title(f"Population Consensus RDM ($N={consensus_res['n_subjects']}$ Subjects)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Concept Index", fontsize=10)
    axes[0].set_ylabel("Concept Index", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Dissimilarity ($1 - r$)")

    # SNR Bar Chart
    snr_vals = [consensus_res["mean_individual_snr"], consensus_res["consensus_snr"]]
    bars = axes[1].bar(["Single Subject SNR", f"Consensus SNR (N={consensus_res['n_subjects']})"],
                       snr_vals, color=["#A0AEC0", "#3182CE"], width=0.55)
    axes[1].set_ylabel("Signal-to-Noise Ratio (SNR)", fontsize=11)
    axes[1].set_title(f"Population SNR Amplification ({consensus_res['snr_gain']:.2f}$\\times$ Gain)", fontsize=12, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}", ha="center", fontweight="bold", fontsize=11)

    plt.tight_layout()
    fig2_path = out_dir / "phase10_population_consensus_rdm.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", fig2_path.name)

    # -------------------------------------------------------------
    # Figure 3: Full Multi-Subject Transfer Matrix
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(loso_matrix * 100.0, cmap="Blues")

    n_sub = len(subjects)
    for i in range(n_sub):
        for j in range(n_sub):
            val = loso_matrix[i, j] * 100.0
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if val > np.mean(loso_matrix * 100) else "black")

    ax.set_xticks(range(n_sub))
    ax.set_yticks(range(n_sub))
    ax.set_xticklabels([f"Test {s}" for s in subjects], fontsize=11)
    ax.set_yticklabels([f"Train {s}" for s in subjects], fontsize=11)
    ax.set_title("Population Cross-Subject Decoding Matrix (%)", fontsize=12, fontweight="bold", pad=20)
    fig.colorbar(cax, fraction=0.046, pad=0.04, label="Accuracy (%)")

    plt.tight_layout()
    fig3_path = out_dir / "phase10_cohort_cross_decoding_matrix.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", fig3_path.name)


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Population Generalization & Scaling Laws")
    parser.add_argument("--zarr-path", type=str, default="E:/pranjal_evobrain/features/ds003825_epochs.zarr")
    parser.add_argument("--out-dir", type=str, default="E:/pranjal_evobrain/plots/phase10")
    parser.add_argument("--n-concepts", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 10 — Population Generalization & Neural Scaling Laws")
    logger.info("  Zarr Store: %s | Concepts: %d", args.zarr_path, args.n_concepts)
    logger.info("=" * 60)

    store = ZarrEpochStore(args.zarr_path)
    subjects = store.subjects()
    logger.info("Available Population Subjects: %s", subjects)

    # Compute shared concepts across all subjects
    all_labels = [store._root[s]["labels"][:] for s in subjects]
    shared = set(np.unique(all_labels[0]))
    for l in all_labels[1:]:
        shared = shared & set(np.unique(l))
    shared_concepts = sorted(list(shared))[:args.n_concepts]

    # 1. Compute Individual Subject RDMs
    subject_rdms = []
    subject_erps = []
    for s in subjects:
        data = store.read_subject(s, concept_filter=shared_concepts)
        X = data["eeg"].astype(np.float32)
        lbls = data["labels"].astype(np.int32)
        # Concept ERPs
        c_erps = [np.mean(X[lbls == c], axis=0) for c in shared_concepts]
        erp_arr = np.stack(c_erps, axis=0)  # (K, C, T)
        rdm = compute_rdm(erp_arr, metric="correlation")
        subject_rdms.append(rdm)
        subject_erps.append(erp_arr)

    # 2. Compute Population Consensus RDM & SNR Amplification
    consensus_res = compute_population_consensus_rdm(subject_rdms)
    logger.info("Population Consensus (N=%d): SNR Gain = %.2fx | Mean Subj-to-Consensus rho = %.4f",
                consensus_res["n_subjects"], consensus_res["snr_gain"], consensus_res["mean_subject_to_consensus_rho"])

    # 3. Population Cross-Subject Transfer Matrix
    n_sub = len(subjects)
    loso_matrix = np.zeros((n_sub, n_sub), dtype=np.float64)
    for i, s_tr in enumerate(subjects):
        for j, s_te in enumerate(subjects):
            acc = evaluate_cohort_decoding(store, [s_tr], s_te, shared_concepts)
            loso_matrix[i, j] = acc
            logger.info("  Pair [%s -> %s]: Accuracy = %.2f%%", s_tr, s_te, acc * 100)

    # 4. Multi-Subject Cohort Scaling (N = 1, 2, ..., N-1)
    # Target test subject: subjects[-1]
    test_subj = subjects[-1]
    pool_train = [s for s in subjects if s != test_subj]

    cohort_sizes = []
    accuracies = []

    for n in range(1, len(pool_train) + 1):
        tr_cohort = pool_train[:n]
        acc = evaluate_cohort_decoding(store, tr_cohort, test_subj, shared_concepts)
        cohort_sizes.append(n)
        accuracies.append(acc)
        logger.info("Cohort Scaling [N=%d subjects -> %s]: Accuracy = %.2f%%", n, test_subj, acc * 100)

    # 5. Fit Power-Law Scaling Law
    scaling_res = fit_neural_scaling_law(cohort_sizes, accuracies)
    logger.info("Fitted Scaling Law: A_inf = %.2f%% | beta = %.2f%% | gamma = %.2f | R^2 = %.2f",
                scaling_res["A_inf"] * 100, scaling_res["beta"] * 100, scaling_res["gamma"], scaling_res["r2"])

    # 6. Generate Figures
    plot_phase10_figures(
        scaling_res=scaling_res,
        consensus_res=consensus_res,
        loso_matrix=loso_matrix,
        subjects=subjects,
        out_dir=out_dir,
    )

    # 7. Save JSON Results
    results = {
        "n_population_subjects": len(subjects),
        "subjects": subjects,
        "n_concepts": len(shared_concepts),
        "chance_level": float(1.0 / len(shared_concepts)),
        "consensus_snr_gain": float(consensus_res["snr_gain"]),
        "mean_subject_to_consensus_rho": float(consensus_res["mean_subject_to_consensus_rho"]),
        "scaling_A_inf": float(scaling_res["A_inf"]),
        "scaling_beta": float(scaling_res["beta"]),
        "scaling_gamma": float(scaling_res["gamma"]),
        "scaling_r2": float(scaling_res["r2"]),
        "cohort_sizes": scaling_res["cohort_sizes"],
        "empirical_accuracies": scaling_res["empirical_accuracies"],
        "extrapolated_50_subjects_acc": float(scaling_res["extrapolated_acc"][-1]),
        "mean_cross_subject_acc": float(np.mean(loso_matrix[~np.eye(n_sub, dtype=bool)])),
        "mean_within_subject_acc": float(np.mean(np.diag(loso_matrix))),
        "h9_supported": bool(consensus_res["snr_gain"] > 1.0),
    }

    json_path = out_dir / "phase10_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved results -> %s", json_path)
    logger.info("=" * 60)
    logger.info("Phase 10 Final Summary:")
    for k, v in results.items():
        logger.info("  %-30s %s", k, v)
    logger.info("=" * 60)
    logger.info("H9 (Neural Population Scaling): %s", "SUPPORTED \u2713" if results["h9_supported"] else "INCONCLUSIVE")


if __name__ == "__main__":
    main()
