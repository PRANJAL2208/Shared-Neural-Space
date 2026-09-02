"""
Phase 5: Population Scaling & Leave-One-Subject-Out (LOSO) Cross-Subject Decoding.

Hypotheses H3 & H4:
    H3: A classifier trained on Subject A (or N-1 subjects) can decode perceived
        visual concepts from an unseen Subject B with accuracy above chance (p < 0.05).
    H4: Cross-subject decodability is driven by shared semantic/perceptual features
        concentrated in the 100–350 ms post-stimulus window.

Evaluation Schemes
------------------
1. Within-Subject 5-Fold Cross-Validation (Baseline Upper Bound).
2. Pairwise Subject-to-Subject Zero-Shot Cross-Decoding (N x N Transfer Matrix).
3. Leave-One-Subject-Out (LOSO) Population Decoding (Train on N-1, test on held-out).
4. Time-Resolved Cross-Subject Decoding Trajectory across [-200 ms, +800 ms].
5. Top-K Concept Nearest-Prototype Retrieval & Permutation Test (N=1000).

Usage
-----
    $env:PYTHONPATH = "."
    python scripts/phase5_loso_decoding.py --n-concepts 50 --n-perms 1000
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

from src.data.zarr_store import ZarrEpochStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase5")

ZARR_PATH    = "E:/pranjal_evobrain/features/ds003825_epochs.zarr"
PLOT_DIR     = Path("E:/pranjal_evobrain/plots/phase5")
N_CONCEPTS   = 50
N_PERMS      = 1000
TMIN_S       = -0.2
SFREQ        = 250.0
WINDOW_MS    = 50


def load_subject_data(store: ZarrEpochStore, subject: str, concepts: list[int]):
    """Load trials belonging to the selected concepts for a given subject."""
    data = store.read_subject(subject)
    X_raw  = data["eeg"].astype(np.float32)       # [n_trials, 63, 251]
    labels = data["labels"].astype(np.int32)     # [n_trials]

    # Filter to selected concepts
    mask = np.isin(labels, concepts)
    X_filtered = X_raw[mask]
    y_filtered = labels[mask]

    # Remap concept IDs to 0..N-1 contiguous indices
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    y_mapped = np.array([concept_to_idx[c] for c in y_filtered], dtype=np.int32)

    logger.info("Loaded %s: %d trials across %d concepts", subject, len(y_mapped), len(concepts))
    return X_filtered, y_mapped


def extract_window_features(X: np.ndarray, tmin_win: float = 0.10, tmax_win: float = 0.35):
    """Extract flattened features in a specific time window [tmin_win, tmax_win]."""
    idx_start = max(0, int((tmin_win - TMIN_S) * SFREQ))
    idx_end   = min(X.shape[-1], int((tmax_win - TMIN_S) * SFREQ))
    X_win = X[:, :, idx_start:idx_end]  # [n_trials, n_ch, n_time_win]
    n_trials = X.shape[0]
    return X_win.reshape(n_trials, -1)


def evaluate_within_subject(X_feat: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """5-Fold Stratified Cross-Validation within a single subject."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_feat, y):
        X_tr, y_tr = X_feat[train_idx], y[train_idx]
        X_te, y_te = X_feat[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = RidgeClassifier(alpha=100.0)
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_te_s)
        accs.append(accuracy_score(y_te, preds))

    return float(np.mean(accs)), float(np.std(accs))


def evaluate_cross_subject_pair(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray):
    """Train on Subject A, evaluate directly on Subject B."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = RidgeClassifier(alpha=100.0)
    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_te_s)
    acc = float(accuracy_score(y_te, preds))
    return acc, clf


def compute_topk_retrieval(erps_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, k_values=(1, 5, 10, 20)):
    """Evaluate Top-K Nearest Prototype Retrieval for unseen subject trials."""
    # erps_train: [n_concepts, n_features]
    # X_test: [n_test_trials, n_features]
    # Compute correlation distance between each test trial and all concept prototypes
    dists = cdist(X_test, erps_train, metric="correlation")  # [n_test_trials, n_concepts]
    rankings = np.argsort(dists, axis=1)  # indices sorted by distance ascending

    topk_accs = {}
    n_trials = len(y_test)
    for k in k_values:
        hits = 0
        for i in range(n_trials):
            if y_test[i] in rankings[i, :k]:
                hits += 1
        topk_accs[k] = float(hits / n_trials)

    return topk_accs, rankings


def time_resolved_decoding(
    subjects_data: dict[str, tuple[np.ndarray, np.ndarray]],
    window_ms: float = 50.0,
    sfreq: float = 250.0,
):
    """Compute time-resolved within-subject and cross-subject decoding trajectory."""
    subj_list = list(subjects_data.keys())
    s_a, s_b = subj_list[0], subj_list[1]
    Xa, ya = subjects_data[s_a]
    Xb, yb = subjects_data[s_b]

    n_time = Xa.shape[-1]
    win_samples = max(1, int(window_ms / 1000.0 * sfreq))

    times = []
    within_accs = []
    cross_accs = []

    for t in range(0, n_time - win_samples + 1, 2):  # step by 2 samples (8 ms)
        sl = slice(t, t + win_samples)
        Fa = Xa[:, :, sl].reshape(len(Xa), -1)
        Fb = Xb[:, :, sl].reshape(len(Xb), -1)

        # 1. Within-subject CV (Subject A)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        w_acc_folds = []
        for tr_idx, te_idx in skf.split(Fa, ya):
            scaler = StandardScaler()
            Fa_tr = scaler.fit_transform(Fa[tr_idx])
            Fa_te = scaler.transform(Fa[te_idx])
            clf = RidgeClassifier(alpha=100.0)
            clf.fit(Fa_tr, ya[tr_idx])
            w_acc_folds.append(accuracy_score(ya[te_idx], clf.predict(Fa_te)))
        within_accs.append(float(np.mean(w_acc_folds)))

        # 2. Cross-subject transfer (Train A -> Test B)
        scaler_cross = StandardScaler()
        Fa_s = scaler_cross.fit_transform(Fa)
        Fb_s = scaler_cross.transform(Fb)
        clf_c = RidgeClassifier(alpha=100.0)
        clf_c.fit(Fa_s, ya)
        cross_accs.append(float(accuracy_score(yb, clf_c.predict(Fb_s))))

        center_t = TMIN_S + (t + win_samples / 2.0) / sfreq
        times.append(center_t)

    return np.array(times), np.array(within_accs), np.array(cross_accs)


def run_permutation_null(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    n_perms: int = 1000,
    random_state: int = 42,
):
    """Permutation test for cross-subject decoding significance."""
    rng = np.random.default_rng(random_state)
    obs_acc, _ = evaluate_cross_subject_pair(X_tr, y_tr, X_te, y_te)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = RidgeClassifier(alpha=100.0)
    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_te_s)

    null_accs = np.empty(n_perms, dtype=np.float32)
    for i in range(n_perms):
        y_te_perm = rng.permutation(y_te)
        null_accs[i] = accuracy_score(y_te_perm, preds)

    p_val = float((np.sum(null_accs >= obs_acc) + 1) / (n_perms + 1))
    return obs_acc, p_val, null_accs


def plot_phase5_figures(
    transfer_matrix: np.ndarray,
    subject_names: list[str],
    times: np.ndarray,
    within_accs: np.ndarray,
    cross_accs: np.ndarray,
    topk_accs: dict[int, float],
    n_concepts: int,
    obs_acc: float,
    p_val: float,
    null_accs: np.ndarray,
    out_dir: Path,
):
    """Generate Phase 5 publication figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chance = 1.0 / n_concepts

    # ── Figure 1: Transfer Matrix & Null Distribution ────────────────────────
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")

    # Heatmap
    im = ax1.imshow(transfer_matrix * 100.0, cmap="viridis", aspect="auto")
    ax1.set_xticks(range(len(subject_names)))
    ax1.set_yticks(range(len(subject_names)))
    ax1.set_xticklabels(subject_names, color="white", fontsize=9)
    ax1.set_yticklabels(subject_names, color="white", fontsize=9)
    ax1.set_title("Cross-Subject Decoding Transfer Matrix (% Correct)\n[Diagonal = Within-Subject CV | Off-Diagonal = Zero-Shot Transfer]",
                  color="white", fontsize=10)
    ax1.set_xlabel("Test Subject", color="white", fontsize=9)
    ax1.set_ylabel("Train Subject", color="white", fontsize=9)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values(): spine.set_edgecolor("#333")

    # Annotate numbers
    for i in range(len(subject_names)):
        for j in range(len(subject_names)):
            val = transfer_matrix[i, j] * 100.0
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center",
                     color="white" if val < (transfer_matrix.max()*80) else "black",
                     fontweight="bold", fontsize=10)
    cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Accuracy (%)", color="white")

    # Permutation Null
    ax2.set_facecolor("#1a1d27")
    ax2.hist(null_accs * 100.0, bins=40, color="#4a90d9", alpha=0.7, edgecolor="none", label="Permutation Null")
    ax2.axvline(chance * 100.0, color="#ff6b6b", lw=1.5, ls="--", label=f"Theoretical Chance ({chance*100:.1f}%)")
    ax2.axvline(obs_acc * 100.0, color="#00ff88", lw=2.2, label=f"Observed Zero-Shot ({obs_acc*100:.2f}%)")
    ax2.set_title(f"Zero-Shot Permutation Significance (N={len(null_accs)})\np = {p_val:.4f}{'*' if p_val < 0.05 else ''}",
                  color="#00ff88" if p_val < 0.05 else "#ff6b6b", fontsize=10)
    ax2.set_xlabel("Decoding Accuracy (%)", color="white", fontsize=9)
    ax2.set_ylabel("Permutation Count", color="white", fontsize=9)
    ax2.tick_params(colors="white")
    ax2.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor("#333")

    fig1.suptitle(f"Phase 5 — Population Decoding & Zero-Shot Generalisation ({n_concepts} Concepts)",
                  color="white", fontsize=12, y=1.02)
    fig1.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase5_loso_transfer_matrix.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig1)
    logger.info("Saved: phase5_loso_transfer_matrix.png")

    # ── Figure 2: Time-Resolved Decoding Trajectory ──────────────────────────
    fig2, ax_t = plt.subplots(figsize=(12, 5), facecolor="#0f1117")
    ax_t.set_facecolor("#1a1d27")
    times_ms = times * 1000.0

    ax_t.plot(times_ms, within_accs * 100.0, color="#4a90d9", lw=2.0, label="Within-Subject (sub-01 CV)")
    ax_t.plot(times_ms, cross_accs * 100.0, color="#00ff88", lw=2.2, label="Zero-Shot Cross-Transfer (sub-01 → sub-02)")
    ax_t.axhline(chance * 100.0, color="#ff6b6b", lw=1.5, ls="--", label=f"Chance Level ({chance*100:.1f}%)")
    ax_t.axvline(0, color="#888888", lw=1.2, ls=":")
    ax_t.axvspan(0, 50, color="#aaaaaa", alpha=0.1, label="RSVP Stimulus (50ms)")

    # Peak annotation
    peak_idx = int(np.argmax(cross_accs))
    peak_t = times_ms[peak_idx]
    peak_val = cross_accs[peak_idx] * 100.0
    ax_t.scatter([peak_t], [peak_val], color="#00ff88", s=50, zorder=5)
    ax_t.annotate(f"Peak Cross-Transfer: {peak_val:.1f}%\n@ {peak_t:.0f} ms",
                  xy=(peak_t, peak_val), xytext=(peak_t + 30, peak_val * 0.95),
                  color="white", fontsize=9,
                  arrowprops=dict(arrowstyle="->", color="#00ff88", lw=1.2))

    ax_t.set_xlabel("Time (ms post-stimulus)", color="white", fontsize=11)
    ax_t.set_ylabel("Decoding Accuracy (% Correct)", color="white", fontsize=11)
    ax_t.set_title(f"Time-Resolved Decoding Dynamics ({n_concepts}-way Concept Classification)",
                   color="white", fontsize=12)
    ax_t.tick_params(colors="white")
    ax_t.legend(frameon=False, labelcolor="white", fontsize=9)
    for spine in ax_t.spines.values(): spine.set_edgecolor("#333")
    fig2.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase5_time_resolved_decoding.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig2)
    logger.info("Saved: phase5_time_resolved_decoding.png")

    # ── Figure 3: Top-K Retrieval Curve ──────────────────────────────────────
    fig3, ax_k = plt.subplots(figsize=(8, 5), facecolor="#0f1117")
    ax_k.set_facecolor("#1a1d27")
    ks = list(topk_accs.keys())
    acc_vals = [topk_accs[k] * 100.0 for k in ks]
    chance_ks = [k / n_concepts * 100.0 for k in ks]

    ax_k.plot(ks, acc_vals, color="#00ff88", marker="o", lw=2.2, label="Cross-Subject Prototype Retrieval")
    ax_k.plot(ks, chance_ks, color="#ff6b6b", ls="--", lw=1.5, label="Random Guess Chance")
    ax_k.fill_between(ks, chance_ks, acc_vals, color="#00ff88", alpha=0.15)

    for k, val in zip(ks, acc_vals):
        ax_k.annotate(f"Top-{k}: {val:.1f}%", xy=(k, val), xytext=(k, val + 2.5),
                      color="white", fontsize=9, ha="center")

    ax_k.set_xlabel("Top-K Candidate Pool", color="white", fontsize=11)
    ax_k.set_ylabel("Retrieval Accuracy (%)", color="white", fontsize=11)
    ax_k.set_title(f"Zero-Shot Top-K Concept Semantic Retrieval (N={n_concepts} Concepts)",
                   color="white", fontsize=12)
    ax_k.set_xticks(ks)
    ax_k.tick_params(colors="white")
    ax_k.legend(frameon=False, labelcolor="white", fontsize=9)
    for spine in ax_k.spines.values(): spine.set_edgecolor("#333")
    fig3.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase5_topk_retrieval_and_null.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig3)
    logger.info("Saved: phase5_topk_retrieval_and_null.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr",       default=ZARR_PATH)
    parser.add_argument("--n-concepts", type=int, default=N_CONCEPTS)
    parser.add_argument("--n-perms",    type=int, default=N_PERMS)
    parser.add_argument("--out-dir",    default=str(PLOT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    store = ZarrEpochStore(args.zarr)
    subjects = store.subjects()

    logger.info("=" * 60)
    logger.info("Phase 5 — Population Decoding & Leave-One-Subject-Out (LOSO)")
    logger.info("  Subjects available: %s | %d concepts | %d perms",
                subjects, args.n_concepts, args.n_perms)
    logger.info("=" * 60)

    # 1. Determine shared concepts across available subjects
    labels_list = [store.read_subject(s)["labels"] for s in subjects]
    shared = set(np.unique(labels_list[0]))
    for l in labels_list[1:]:
        shared = shared & set(np.unique(l))

    # Pick top n concepts
    concepts = sorted(list(shared))[:args.n_concepts]
    logger.info("Selected %d common concepts: %s", len(concepts), concepts[:5])

    # 2. Load trial data & extract core visual window features [100ms - 350ms]
    subjects_raw = {}
    subjects_feat = {}
    for s in subjects:
        X_raw, y = load_subject_data(store, s, concepts)
        subjects_raw[s] = (X_raw, y)
        subjects_feat[s] = (extract_window_features(X_raw, tmin_win=0.10, tmax_win=0.35), y)

    # 3. Compute N x N Transfer Matrix
    n_subj = len(subjects)
    transfer_matrix = np.zeros((n_subj, n_subj), dtype=np.float32)

    for i, s_tr in enumerate(subjects):
        for j, s_te in enumerate(subjects):
            if i == j:
                # Within-subject 5-fold CV
                acc, _ = evaluate_within_subject(subjects_feat[s_tr][0], subjects_feat[s_tr][1])
                transfer_matrix[i, j] = acc
                logger.info("  Within-subject [%s]: %.2f%%", s_tr, acc * 100.0)
            else:
                # Cross-subject transfer
                acc, _ = evaluate_cross_subject_pair(
                    subjects_feat[s_tr][0], subjects_feat[s_tr][1],
                    subjects_feat[s_te][0], subjects_feat[s_te][1],
                )
                transfer_matrix[i, j] = acc
                logger.info("  Cross-transfer [%s → %s]: %.2f%%", s_tr, s_te, acc * 100.0)

    # 4. Zero-shot Permutation Null (sub-01 -> sub-02)
    s1, s2 = subjects[0], subjects[1]
    logger.info("Computing permutation null test (%s → %s, N=%d) …", s1, s2, args.n_perms)
    obs_acc, p_val, null_accs = run_permutation_null(
        subjects_feat[s1][0], subjects_feat[s1][1],
        subjects_feat[s2][0], subjects_feat[s2][1],
        n_perms=args.n_perms,
    )
    logger.info("  Observed Zero-Shot Acc: %.2f%% | p-value: %.4f %s",
                obs_acc * 100.0, p_val, "*** SIGNIFICANT" if p_val < 0.05 else "")

    # 5. Top-K Semantic Concept Prototype Retrieval
    # Compute prototypes on training subject (sub-01)
    X_s1, y_s1 = subjects_feat[s1]
    X_s2, y_s2 = subjects_feat[s2]
    prototypes_s1 = np.stack([X_s1[y_s1 == c].mean(axis=0) for c in range(args.n_concepts)], axis=0)
    topk_accs, _ = compute_topk_retrieval(prototypes_s1, X_s2, y_s2, k_values=(1, 5, 10, 20))
    logger.info("Zero-Shot Top-K Retrieval on %s:", s2)
    for k, vacc in topk_accs.items():
        logger.info("  Top-%-2d Retrieval: %.2f%% (Chance: %.2f%%)", k, vacc * 100.0, (k / args.n_concepts) * 100.0)

    # 6. Time-Resolved Cross-Subject Decoding
    logger.info("Computing time-resolved decoding trajectory (%dms window) …", WINDOW_MS)
    times, within_accs, cross_accs = time_resolved_decoding(
        subjects_raw, window_ms=WINDOW_MS, sfreq=SFREQ
    )
    peak_t_idx = int(np.argmax(cross_accs))
    peak_cross_t = float(times[peak_t_idx] * 1000.0)
    peak_cross_acc = float(cross_accs[peak_t_idx])
    logger.info("  Peak Cross-Subject Decoding: %.2f%% at %.0f ms post-stimulus",
                peak_cross_acc * 100.0, peak_cross_t)

    # 7. Generate Figures
    logger.info("Generating publication figures → %s", out_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_phase5_figures(
            transfer_matrix, subjects,
            times, within_accs, cross_accs,
            topk_accs, args.n_concepts,
            obs_acc, p_val, null_accs,
            out_dir,
        )

    # 8. Save structured JSON
    chance_level = 1.0 / args.n_concepts
    mean_within = float(np.mean(np.diag(transfer_matrix)))
    off_diag_mask = ~np.eye(n_subj, dtype=bool)
    mean_cross = float(np.mean(transfer_matrix[off_diag_mask])) if off_diag_mask.any() else obs_acc

    results = {
        "n_subjects": n_subj,
        "subjects": subjects,
        "n_concepts": args.n_concepts,
        "chance_level": round(chance_level, 6),
        "mean_within_subject_acc": round(mean_within, 6),
        "mean_cross_subject_acc": round(mean_cross, 6),
        "zero_shot_pair_acc": round(obs_acc, 6),
        "permutation_p_value": round(p_val, 6),
        "significant_p05": p_val < 0.05,
        "peak_decoding_latency_ms": round(peak_cross_t, 1),
        "peak_decoding_acc": round(peak_cross_acc, 6),
        "top_1_retrieval_acc": round(topk_accs[1], 6),
        "top_5_retrieval_acc": round(topk_accs[5], 6),
        "top_10_retrieval_acc": round(topk_accs[10], 6),
        "top_20_retrieval_acc": round(topk_accs[20], 6),
        "h3_supported": p_val < 0.05 and (obs_acc > chance_level),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase5_results.json").write_text(json.dumps(results, indent=2))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 5 Final Results Summary")
    logger.info("=" * 60)
    for k, v in results.items():
        if k not in ("subjects",):
            logger.info("  %-32s %s", k, v)
    logger.info("=" * 60)
    logger.info("H₃ (Cross-Subject Zero-Shot Decoding): %s",
                "SUPPORTED ✓" if results["h3_supported"] else "NOT SUPPORTED ✗")
    logger.info("All plots saved to: %s", out_dir)


if __name__ == "__main__":
    main()
