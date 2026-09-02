"""
Phase 6: Subject-Invariant Contrastive Latent Alignment Architecture & Training.

Hypotheses H3, H4, & H5 (Stage 2 Latent Alignment):
    H4: Training a neural encoder with cross-subject Supervised Contrastive Loss
        pulls single-trial neural representations of the same visual concept
        together across individuals while suppressing subject-specific noise.
    H5: Latent representations Z achieve higher zero-shot concept classification
        and retrieval accuracy on an unseen subject than raw sensor-space features.

Pipeline
--------
1. Load trials from training subjects (sub-01, sub-02) and held-out subject (sub-03).
2. Train EEGNetEncoder using SupConLoss with concept-balanced multi-subject batching.
3. Monitor probe metrics (Concept Decodability vs Subject Invariance).
4. Project test trials from held-out subject into latent space Z.
5. Quantify zero-shot Top-K retrieval, linear probing, and latent RDM alignment.
6. Generate publication figures and serialize model weights and metrics.

Usage
-----
    $env:PYTHONPATH = "."
    python scripts/phase6_train_contrastive_alignment.py --epochs 25 --n-concepts 50
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data.zarr_store import ZarrEpochStore
from src.models import (
    EEGNetEncoder,
    SupConLoss,
    MultiSubjectEEGDataset,
    ConceptBalancedBatchSampler,
)
from src.evaluation.probes import LinearProber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase6")

ZARR_PATH    = "E:/pranjal_evobrain/features/ds003825_epochs.zarr"
PLOT_DIR     = Path("E:/pranjal_evobrain/plots/phase6")
CKPT_DIR     = Path("E:/pranjal_evobrain/checkpoints")
N_CONCEPTS   = 50
EPOCHS       = 25
BATCH_CONCEPTS = 16
SAMPLES_PER_CONCEPT = 4
LR           = 1e-3
LATENT_DIM   = 128
PROJECTION_DIM = 128


def load_dataset(store: ZarrEpochStore, train_subjs: list[str], test_subj: str, n_concepts: int):
    """Load and format multi-subject training and testing datasets."""
    all_subjs = train_subjs + [test_subj]
    labels_list = [store.read_subject(s)["labels"] for s in all_subjs]

    # Find shared concepts
    shared = set(np.unique(labels_list[0]))
    for l in labels_list[1:]:
        shared = shared & set(np.unique(l))

    concepts = sorted(list(shared))[:n_concepts]
    concept_to_idx = {c: i for i, c in enumerate(concepts)}

    # Training data
    X_train_list, y_train_list, s_train_list = [], [], []
    for s_idx, s in enumerate(train_subjs):
        data = store.read_subject(s)
        X_raw = data["eeg"].astype(np.float32)
        lbls = data["labels"].astype(np.int32)
        mask = np.isin(lbls, concepts)
        X_sub = X_raw[mask]
        y_sub = np.array([concept_to_idx[c] for c in lbls[mask]], dtype=np.int32)
        s_sub = np.full(len(y_sub), s_idx, dtype=np.int32)

        X_train_list.append(X_sub)
        y_train_list.append(y_sub)
        s_train_list.append(s_sub)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    s_train = np.concatenate(s_train_list, axis=0)

    # Test data (held-out subject)
    data_test = store.read_subject(test_subj)
    X_raw_te = data_test["eeg"].astype(np.float32)
    lbls_te = data_test["labels"].astype(np.int32)
    mask_te = np.isin(lbls_te, concepts)
    X_test = X_raw_te[mask_te]
    y_test = np.array([concept_to_idx[c] for c in lbls_te[mask_te]], dtype=np.int32)
    s_test = np.full(len(y_test), len(train_subjs), dtype=np.int32)

    logger.info("Dataset prepared: Train=%d trials across %s | Test (%s)=%d trials | %d concepts",
                len(X_train), train_subjs, test_subj, len(X_test), n_concepts)
    return (X_train, y_train, s_train), (X_test, y_test, s_test), concepts


def train_contrastive_model(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_concepts: int,
    epochs: int = 25,
    device: torch.device = torch.device("cpu"),
):
    """Train EEGNetEncoder with Supervised Contrastive Loss."""
    X_tr, y_tr, s_tr = train_data
    X_te, y_te, s_te = test_data

    dataset = MultiSubjectEEGDataset(X_tr, y_tr, s_tr)
    sampler = ConceptBalancedBatchSampler(
        y_tr,
        n_concepts_per_batch=BATCH_CONCEPTS,
        n_samples_per_concept=SAMPLES_PER_CONCEPT,
        n_batches=max(30, len(X_tr) // (BATCH_CONCEPTS * SAMPLES_PER_CONCEPT)),
    )
    loader = DataLoader(dataset, batch_sampler=sampler)

    n_channels, n_samples = X_tr.shape[1], X_tr.shape[2]
    model = EEGNetEncoder(
        n_channels=n_channels,
        n_samples=n_samples,
        latent_dim=LATENT_DIM,
        projection_dim=PROJECTION_DIM,
        dropout=0.25,
    ).to(device)

    criterion = SupConLoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "loss": [],
        "concept_probe_acc": [],
        "subject_probe_acc": [],
        "zero_shot_test_acc": [],
    }

    logger.info("Starting Contrastive Training (%d epochs) …", epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for x_batch, c_batch, s_batch in loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)

            optimizer.zero_grad()
            _, z_proj = model(x_batch)
            loss = criterion(z_proj, c_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses))
        history["loss"].append(avg_loss)

        # Evaluate Probes every 5 epochs
        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            model.eval()
            with torch.no_grad():
                Z_tr_eval = model.encode(torch.tensor(X_tr, dtype=torch.float32).to(device)).cpu().numpy()
                Z_te_eval = model.encode(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()

            prober = LinearProber(max_iter=500, random_state=42)
            prober.fit(Z_tr_eval, y_tr, s_tr)
            probe_res = prober.evaluate(Z_tr_eval, y_tr, s_tr)

            # Zero-shot test accuracy on held-out subject
            clf_zero = RidgeClassifier(alpha=10.0)
            clf_zero.fit(Z_tr_eval, y_tr)
            zero_acc = float(accuracy_score(y_te, clf_zero.predict(Z_te_eval)))

            history["concept_probe_acc"].append(probe_res["concept_accuracy"])
            history["subject_probe_acc"].append(probe_res["subject_accuracy"])
            history["zero_shot_test_acc"].append(zero_acc)

            logger.info(
                "Epoch %02d/%02d | Loss: %.4f | Concept Probe: %.2f%% | Subject Probe: %.2f%% | Zero-Shot Test: %.2f%%",
                epoch, epochs, avg_loss,
                probe_res["concept_accuracy"] * 100.0,
                probe_res["subject_accuracy"] * 100.0,
                zero_acc * 100.0,
            )

    return model, history


def evaluate_latent_representations(
    model: EEGNetEncoder,
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_concepts: int,
    device: torch.device,
):
    """Perform thorough post-alignment zero-shot evaluation on latent embeddings."""
    X_tr, y_tr, s_tr = train_data
    X_te, y_te, s_te = test_data

    model.eval()
    with torch.no_grad():
        Z_tr = model.encode(torch.tensor(X_tr, dtype=torch.float32).to(device)).cpu().numpy()
        Z_te = model.encode(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()

    # 1. Linear Probe Evaluation
    prober = LinearProber(max_iter=1000, random_state=42)
    prober.fit(Z_tr, y_tr, s_tr)
    probe_metrics = prober.evaluate(Z_tr, y_tr, s_tr)

    # 2. Zero-Shot Concept Classifier on Held-Out Subject
    clf = RidgeClassifier(alpha=10.0)
    clf.fit(Z_tr, y_tr)
    zero_shot_acc = float(accuracy_score(y_te, clf.predict(Z_te)))

    # 3. Latent Prototype Top-K Retrieval
    prototypes = np.stack([Z_tr[y_tr == c].mean(axis=0) for c in range(n_concepts)], axis=0)
    dists = cdist(Z_te, prototypes, metric="cosine")
    rankings = np.argsort(dists, axis=1)

    topk_accs = {}
    for k in (1, 5, 10, 20):
        hits = sum(1 for i in range(len(y_te)) if y_te[i] in rankings[i, :k])
        topk_accs[k] = float(hits / len(y_te))

    # 4. Latent PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    Z_all = np.concatenate([Z_tr, Z_te], axis=0)
    Z_pca = pca.fit_transform(Z_all)
    Z_tr_pca = Z_pca[:len(Z_tr)]
    Z_te_pca = Z_pca[len(Z_tr):]

    return {
        "Z_train": Z_tr,
        "Z_test": Z_te,
        "Z_train_pca": Z_tr_pca,
        "Z_test_pca": Z_te_pca,
        "zero_shot_acc": zero_shot_acc,
        "topk_accs": topk_accs,
        "probe_metrics": probe_metrics,
    }


def plot_phase6_figures(
    history: dict,
    eval_results: dict,
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_concepts: int,
    out_dir: Path,
):
    """Generate dark-mode publication figures for Phase 6."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chance = 1.0 / n_concepts
    y_tr, s_tr = train_data[1], train_data[2]
    y_te = test_data[1]

    # ── Figure 1: Training Loss & Linear Probe Trajectories ──────────────────
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")

    # Loss Curve
    ax1.set_facecolor("#1a1d27")
    ax1.plot(range(1, len(history["loss"]) + 1), history["loss"], color="#00ff88", lw=2.2, label="InfoNCE Loss")
    ax1.set_xlabel("Epoch", color="white", fontsize=11)
    ax1.set_ylabel("Contrastive Loss", color="white", fontsize=11)
    ax1.set_title("Supervised Contrastive Optimization (SupCon)", color="white", fontsize=12)
    ax1.tick_params(colors="white")
    ax1.legend(frameon=False, labelcolor="white", fontsize=9)
    for spine in ax1.spines.values(): spine.set_edgecolor("#333")

    # Probing Metrics
    ax2.set_facecolor("#1a1d27")
    eval_epochs = [1] + list(range(5, len(history["loss"]) + 1, 5))
    if eval_epochs[-1] != len(history["loss"]):
        eval_epochs.append(len(history["loss"]))

    c_accs = [acc * 100.0 for acc in history["concept_probe_acc"]]
    s_accs = [acc * 100.0 for acc in history["subject_probe_acc"]]
    z_accs = [acc * 100.0 for acc in history["zero_shot_test_acc"]]

    ax2.plot(eval_epochs, c_accs, color="#4a90d9", marker="o", lw=2.0, label="Concept Probe (Train Subjs)")
    ax2.plot(eval_epochs, z_accs, color="#00ff88", marker="s", lw=2.2, label="Zero-Shot Test (Held-Out Subj)")
    ax2.plot(eval_epochs, s_accs, color="#ff6b6b", marker="^", lw=1.8, ls=":", label="Subject Probe (Invariance Target)")
    ax2.axhline(chance * 100.0, color="#888888", lw=1.2, ls="--", label=f"Concept Chance ({chance*100:.1f}%)")

    ax2.set_xlabel("Epoch", color="white", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", color="white", fontsize=11)
    ax2.set_title("Probing Dynamics: Concept Emergence vs Subject Invariance", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    ax2.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor("#333")

    fig1.suptitle("Phase 6 — Contrastive Neural Latent Alignment (InfoNCE Optimization)", color="white", fontsize=12, y=1.02)
    fig1.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase6_training_loss_and_probes.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig1)
    logger.info("Saved: phase6_training_loss_and_probes.png")

    # ── Figure 2: Latent Manifold Projection (Concept vs Subject) ────────────
    fig2, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f1117")

    Z_tr_pca = eval_results["Z_train_pca"]
    Z_te_pca = eval_results["Z_test_pca"]

    # 1. Colored by Subject Identity
    ax_p1.set_facecolor("#1a1d27")
    ax_p1.scatter(Z_tr_pca[s_tr == 0, 0], Z_tr_pca[s_tr == 0, 1], color="#4a90d9", s=18, alpha=0.6, label="sub-01")
    ax_p1.scatter(Z_tr_pca[s_tr == 1, 0], Z_tr_pca[s_tr == 1, 1], color="#e67e22", s=18, alpha=0.6, label="sub-02")
    ax_p1.scatter(Z_te_pca[:, 0], Z_te_pca[:, 1], color="#00ff88", s=22, alpha=0.7, label="sub-03 (Held-Out)")
    ax_p1.set_title("Latent Space Z: Colored by Subject Identity\n(Subjects Intermixed = Invariant)", color="white", fontsize=10)
    ax_p1.set_xlabel("Latent PC 1", color="white", fontsize=9)
    ax_p1.set_ylabel("Latent PC 2", color="white", fontsize=9)
    ax_p1.tick_params(colors="white")
    ax_p1.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax_p1.spines.values(): spine.set_edgecolor("#333")

    # 2. Colored by Concept Categories (Subset of top 8 concepts for clarity)
    ax_p2.set_facecolor("#1a1d27")
    cmap = matplotlib.colormaps.get_cmap("tab10") if hasattr(matplotlib, "colormaps") else plt.get_cmap("tab10")
    for c_i in range(8):
        mask_c = (y_tr == c_i)
        ax_p2.scatter(Z_tr_pca[mask_c, 0], Z_tr_pca[mask_c, 1], color=cmap(c_i), s=25, alpha=0.8, label=f"Concept {c_i}")
        mask_te_c = (y_te == c_i)
        ax_p2.scatter(Z_te_pca[mask_te_c, 0], Z_te_pca[mask_te_c, 1], color=cmap(c_i), s=35, marker="^", edgecolors="white", lw=0.5)

    ax_p2.set_title("Latent Space Z: Colored by Concept Label\n(Circles=Train Subjs, Triangles=Held-Out Subj)", color="white", fontsize=10)
    ax_p2.set_xlabel("Latent PC 1", color="white", fontsize=9)
    ax_p2.set_ylabel("Latent PC 2", color="white", fontsize=9)
    ax_p2.tick_params(colors="white")
    ax_p2.legend(frameon=False, labelcolor="white", fontsize=7, ncol=2)
    for spine in ax_p2.spines.values(): spine.set_edgecolor("#333")

    fig2.suptitle("Phase 6 — Emergent Subject-Invariant Concept Manifold in Latent Space Z", color="white", fontsize=12, y=1.02)
    fig2.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase6_latent_manifold_projections.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig2)
    logger.info("Saved: phase6_latent_manifold_projections.png")

    # ── Figure 3: Performance Gains: Raw Sensor Space vs Latent Space ────────
    fig3, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0f1117")

    # Bar chart comparison
    ax_b1.set_facecolor("#1a1d27")
    raw_acc = 1.83  # from Phase 5 baseline
    latent_acc = eval_results["zero_shot_acc"] * 100.0
    concept_probe_acc = eval_results["probe_metrics"]["concept_accuracy"] * 100.0

    bars = ax_b1.bar(["Chance\nLevel", "Raw Sensor Space\n(Phase 5 Baseline)", "Shared Latent Z\n(Zero-Shot sub-03)", "Shared Latent Z\n(Concept Probe)"],
                     [chance * 100.0, raw_acc, latent_acc, concept_probe_acc],
                     color=["#888888", "#e67e22", "#00ff88", "#4a90d9"],
                     width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax_b1.text(bar.get_x() + bar.get_width()/2.0, h + 0.5, f"{h:.2f}%", ha="center", va="bottom", color="white", fontweight="bold", fontsize=9)

    ax_b1.set_ylabel("Accuracy (% Correct)", color="white", fontsize=10)
    ax_b1.set_title("Zero-Shot Decoding Gain on Held-Out Human (sub-03)", color="white", fontsize=11)
    ax_b1.tick_params(colors="white", labelsize=8)
    for spine in ax_b1.spines.values(): spine.set_edgecolor("#333")

    # Top-K Retrieval Curve Comparison
    ax_b2.set_facecolor("#1a1d27")
    ks = list(eval_results["topk_accs"].keys())
    latent_topk = [eval_results["topk_accs"][k] * 100.0 for k in ks]
    chance_topk = [k / n_concepts * 100.0 for k in ks]

    ax_b2.plot(ks, latent_topk, color="#00ff88", marker="o", lw=2.2, label="Contrastive Latent Z (Held-Out Subj)")
    ax_b2.plot(ks, chance_topk, color="#ff6b6b", ls="--", lw=1.5, label="Random Guess Chance")
    ax_b2.fill_between(ks, chance_topk, latent_topk, color="#00ff88", alpha=0.15)

    for k, v in zip(ks, latent_topk):
        ax_b2.annotate(f"{v:.1f}%", xy=(k, v), xytext=(k, v + 2.5), color="white", fontsize=9, ha="center")

    ax_b2.set_xlabel("Top-K Candidate Pool", color="white", fontsize=10)
    ax_b2.set_ylabel("Retrieval Accuracy (%)", color="white", fontsize=10)
    ax_b2.set_title("Zero-Shot Semantic Prototype Retrieval in Latent Space Z", color="white", fontsize=11)
    ax_b2.set_xticks(ks)
    ax_b2.tick_params(colors="white")
    ax_b2.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax_b2.spines.values(): spine.set_edgecolor("#333")

    fig3.suptitle("Phase 6 — Validation of Hypothesis H4 & H5 (Latent Representation Superiority)", color="white", fontsize=12, y=1.02)
    fig3.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase6_contrastive_transfer_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig3)
    logger.info("Saved: phase6_contrastive_transfer_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr",       default=ZARR_PATH)
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--n-concepts", type=int, default=N_CONCEPTS)
    parser.add_argument("--out-dir",    default=str(PLOT_DIR))
    parser.add_argument("--ckpt-dir",   default=str(CKPT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store = ZarrEpochStore(args.zarr)
    train_subjs = ["sub-01", "sub-02"]
    test_subj = "sub-03"

    logger.info("=" * 60)
    logger.info("Phase 6 — Contrastive Latent Alignment Training (Stage 2)")
    logger.info("  Train Subjects: %s | Test Subject: %s | Epochs: %d | Device: %s",
                train_subjs, test_subj, args.epochs, device)
    logger.info("=" * 60)

    # 1. Prepare Datasets
    train_data, test_data, concepts = load_dataset(store, train_subjs, test_subj, args.n_concepts)

    # 2. Train Contrastive Encoder
    model, history = train_contrastive_model(
        train_data, test_data, args.n_concepts, epochs=args.epochs, device=device
    )

    # 3. Save Model Checkpoint
    ckpt_path = ckpt_dir / "contrastive_eegnet.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_concepts": args.n_concepts,
        "latent_dim": LATENT_DIM,
        "projection_dim": PROJECTION_DIM,
        "train_subjects": train_subjs,
        "test_subject": test_subj,
    }, ckpt_path)
    logger.info("Saved model checkpoint → %s", ckpt_path)

    # 4. Thorough Latent Evaluation
    logger.info("Evaluating latent embeddings on held-out subject (%s) …", test_subj)
    eval_results = evaluate_latent_representations(
        model, train_data, test_data, args.n_concepts, device
    )

    # 5. Generate Figures
    logger.info("Generating publication figures → %s", out_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_phase6_figures(history, eval_results, train_data, test_data, args.n_concepts, out_dir)

    # 6. Structured JSON Results
    chance = 1.0 / args.n_concepts
    results = {
        "train_subjects": train_subjs,
        "test_subject": test_subj,
        "n_concepts": args.n_concepts,
        "epochs_trained": args.epochs,
        "final_contrastive_loss": round(history["loss"][-1], 6),
        "chance_level": round(chance, 6),
        "raw_sensor_baseline_acc": 0.0183,
        "latent_zero_shot_acc": round(eval_results["zero_shot_acc"], 6),
        "concept_probe_acc": round(eval_results["probe_metrics"]["concept_accuracy"], 6),
        "subject_probe_acc": round(eval_results["probe_metrics"]["subject_accuracy"], 6),
        "subject_chance": round(eval_results["probe_metrics"]["subject_chance"], 6),
        "subject_invariance_achieved": eval_results["probe_metrics"]["subject_accuracy"] <= 0.60,
        "top_1_latent_retrieval": round(eval_results["topk_accs"][1], 6),
        "top_5_latent_retrieval": round(eval_results["topk_accs"][5], 6),
        "top_10_latent_retrieval": round(eval_results["topk_accs"][10], 6),
        "top_20_latent_retrieval": round(eval_results["topk_accs"][20], 6),
        "h4_h5_supported": eval_results["zero_shot_acc"] > 0.0183 and eval_results["topk_accs"][1] > chance,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase6_results.json").write_text(json.dumps(results, indent=2))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 6 Final Results Summary")
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info("  %-32s %s", k, v)
    logger.info("=" * 60)
    logger.info("H₄ & H₅ (Contrastive Alignment Superiority): %s",
                "SUPPORTED ✓" if results["h4_h5_supported"] else "NOT SUPPORTED ✗")
    logger.info("All artifacts saved to: %s", out_dir)


if __name__ == "__main__":
    main()
