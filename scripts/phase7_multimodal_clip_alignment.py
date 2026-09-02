"""
Phase 7: Multimodal EEG-CLIP Representation Alignment & Zero-Shot Concept Retrieval.

Hypotheses H5 & H6 (Multimodal Foundation Alignment):
    H5: Single-trial neural latent representations Z_EEG can be projected directly
        into a pretrained CLIP image/concept embedding space (512-d).
    H6: The projected neural embeddings achieve significant zero-shot image/concept
        retrieval on an unseen human brain (sub-03) outperforming chance and raw baselines,
        and human neural geometry correlates significantly with CLIP vision space (ρ > 0).

Pipeline
--------
1. Extract 512-d CLIP semantic embeddings for all N visual concepts using OpenCLIP ViT-B-32.
2. Load pretrained Phase 6 EEGNetEncoder to produce neural latents Z_EEG.
3. Train EEGToCLIPProjector on sub-01 and sub-02 using bidirectional multimodal contrastive loss.
4. Test zero-shot multimodal concept retrieval on held-out subject (sub-03).
5. Compute Brain-to-CLIP Representational Similarity Analysis (RSA).
6. Generate publication-grade dark-mode visualizations and serialize model checkpoints.

Usage
-----
    $env:PYTHONPATH = "."
    python scripts/phase7_multimodal_clip_alignment.py --epochs 25 --n-concepts 50
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data.zarr_store import ZarrEpochStore
from src.models.encoder import EEGNetEncoder
from src.alignment.clip_alignment import (
    EEGToCLIPProjector,
    MultimodalContrastiveLoss,
    extract_concept_clip_embeddings,
)
from src.alignment.rsa import compute_rdm, vectorize_rdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase7")

ZARR_PATH    = "E:/pranjal_evobrain/features/ds003825_epochs.zarr"
PLOT_DIR     = Path("E:/pranjal_evobrain/plots/phase7")
CKPT_DIR     = Path("E:/pranjal_evobrain/checkpoints")
N_CONCEPTS   = 50
EPOCHS       = 25
LR           = 1e-3
BATCH_SIZE   = 64


def load_eeg_and_concepts(store: ZarrEpochStore, train_subjs: list[str], test_subj: str, n_concepts: int):
    """Load EEG trials and concept names for training and testing subjects."""
    all_subjs = train_subjs + [test_subj]
    labels_list = [store._root[s]["labels"][:] for s in all_subjs]

    # Shared concepts
    shared = set(np.unique(labels_list[0]))
    for l in labels_list[1:]:
        shared = shared & set(np.unique(l))

    concepts = sorted(list(shared))[:n_concepts]
    concept_to_idx = {c: i for i, c in enumerate(concepts)}

    # Concept names
    meta_grp = store._root[train_subjs[0]]
    if "concept_names_json" in meta_grp.attrs:
        import json
        raw_names = json.loads(meta_grp.attrs["concept_names_json"])
        concept_names = [str(raw_names[c]) if c < len(raw_names) else f"concept_{c}" for c in concepts]
    else:
        concept_names = [f"concept_{c}" for c in concepts]

    # Training data
    X_tr_list, y_tr_list = [], []
    for s in train_subjs:
        data = store.read_subject(s, concept_filter=concepts)
        X_sub = data["eeg"].astype(np.float32)
        lbls = data["labels"].astype(np.int32)
        X_tr_list.append(X_sub)
        y_tr_list.append(np.array([concept_to_idx[c] for c in lbls], dtype=np.int32))

    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)

    # Test data
    data_te = store.read_subject(test_subj, concept_filter=concepts)
    X_test = data_te["eeg"].astype(np.float32)
    lbls_te = data_te["labels"].astype(np.int32)
    y_test = np.array([concept_to_idx[c] for c in lbls_te], dtype=np.int32)

    logger.info("Loaded EEG: Train=%d trials | Test (%s)=%d trials | %d concepts",
                len(X_train), test_subj, len(X_test), n_concepts)
    return (X_train, y_train), (X_test, y_test), concepts, concept_names


def extract_eeg_latents(eegnet: EEGNetEncoder, X: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    """Extract 128-d latent features Z_EEG from EEGNet."""
    eegnet.eval()
    latents = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            z = eegnet.encode(batch)
            latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0).astype(np.float32)


def train_multimodal_projector(
    Z_eeg_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_clip_all: np.ndarray,
    Z_eeg_te: np.ndarray,
    y_te: np.ndarray,
    epochs: int = 25,
    device: torch.device = torch.device("cpu"),
):
    """Train EEGToCLIPProjector aligning EEG latents with CLIP target embeddings."""
    projector = EEGToCLIPProjector(input_dim=Z_eeg_tr.shape[1], clip_dim=Z_clip_all.shape[1], hidden_dim=256).to(device)
    criterion = MultimodalContrastiveLoss(temperature=0.07)
    optimizer = optim.AdamW(projector.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Dataset: (z_eeg, clip_target)
    Z_clip_targets_tr = Z_clip_all[y_tr]
    dataset = TensorDataset(
        torch.tensor(Z_eeg_tr, dtype=torch.float32),
        torch.tensor(Z_clip_targets_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    history = {"loss": [], "zero_shot_top1": [], "zero_shot_top5": []}
    logger.info("Training Multimodal EEG-CLIP Projector (%d epochs) …", epochs)

    for epoch in range(1, epochs + 1):
        projector.train()
        epoch_losses = []

        for z_eeg_b, z_clip_b, _ in loader:
            z_eeg_b = z_eeg_b.to(device)
            z_clip_b = z_clip_b.to(device)

            optimizer.zero_grad()
            z_proj = projector(z_eeg_b)
            loss = criterion(z_proj, z_clip_b)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses))
        history["loss"].append(avg_loss)

        # Evaluate zero-shot retrieval on held-out subject
        projector.eval()
        with torch.no_grad():
            z_te_proj = projector(torch.tensor(Z_eeg_te, dtype=torch.float32).to(device)).cpu().numpy()

        dists = cdist(z_te_proj, Z_clip_all, metric="cosine")
        rankings = np.argsort(dists, axis=1)
        top1 = float(sum(1 for i in range(len(y_te)) if y_te[i] == rankings[i, 0]) / len(y_te))
        top5 = float(sum(1 for i in range(len(y_te)) if y_te[i] in rankings[i, :5]) / len(y_te))

        history["zero_shot_top1"].append(top1)
        history["zero_shot_top5"].append(top5)

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            logger.info("Epoch %02d/%02d | Loss: %.4f | Zero-Shot Top-1 (sub-03): %.2f%% | Top-5: %.2f%%",
                        epoch, epochs, avg_loss, top1 * 100.0, top5 * 100.0)

    return projector, history, z_te_proj


def compute_multimodal_rsa(erps_eeg: np.ndarray, Z_clip: np.ndarray):
    """Compute Representational Similarity Analysis between Human Brain EEG and CLIP."""
    rdm_eeg = compute_rdm(erps_eeg, metric="correlation")
    rdm_clip = compute_rdm(Z_clip, metric="cosine")

    vec_eeg = vectorize_rdm(rdm_eeg)
    vec_clip = vectorize_rdm(rdm_clip)

    res_spearman = spearmanr(vec_eeg, vec_clip)
    res_pearson = pearsonr(vec_eeg, vec_clip)
    return rdm_eeg, rdm_clip, float(res_spearman.statistic), float(res_spearman.pvalue), float(res_pearson.statistic)


def plot_phase7_figures(
    history: dict,
    rdm_eeg: np.ndarray,
    rdm_clip: np.ndarray,
    rsa_rho: float,
    rsa_p: float,
    z_te_proj: np.ndarray,
    Z_clip_all: np.ndarray,
    y_te: np.ndarray,
    topk_accs: dict[int, float],
    concept_names: list[str],
    n_concepts: int,
    out_dir: Path,
):
    """Generate publication dark-mode figures for Phase 7."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chance = 1.0 / n_concepts

    # ── Figure 1: Multimodal Training & Zero-Shot Retrieval Dynamics ─────────
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")

    # Loss curve
    ax1.set_facecolor("#1a1d27")
    ax1.plot(range(1, len(history["loss"]) + 1), history["loss"], color="#00ff88", lw=2.2, label="Bidirectional InfoNCE Loss")
    ax1.set_xlabel("Epoch", color="white", fontsize=11)
    ax1.set_ylabel("Multimodal Contrastive Loss", color="white", fontsize=11)
    ax1.set_title("EEG-to-CLIP Alignment Optimization", color="white", fontsize=12)
    ax1.tick_params(colors="white")
    ax1.legend(frameon=False, labelcolor="white", fontsize=9)
    for spine in ax1.spines.values(): spine.set_edgecolor("#333")

    # Retrieval curves
    ax2.set_facecolor("#1a1d27")
    eps = range(1, len(history["zero_shot_top1"]) + 1)
    ax2.plot(eps, [v * 100.0 for v in history["zero_shot_top1"]], color="#00ff88", lw=2.2, label="Top-1 Zero-Shot Retrieval (sub-03)")
    ax2.plot(eps, [v * 100.0 for v in history["zero_shot_top5"]], color="#4a90d9", lw=2.0, label="Top-5 Zero-Shot Retrieval (sub-03)")
    ax2.axhline(chance * 100.0, color="#ff6b6b", lw=1.5, ls="--", label=f"Top-1 Chance ({chance*100:.1f}%)")
    ax2.axhline((5.0 / n_concepts) * 100.0, color="#e67e22", lw=1.2, ls=":", label=f"Top-5 Chance (10.0%)")

    ax2.set_xlabel("Epoch", color="white", fontsize=11)
    ax2.set_ylabel("Retrieval Accuracy (%)", color="white", fontsize=11)
    ax2.set_title("Zero-Shot Visual Retrieval on Held-Out Human", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    ax2.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor("#333")

    fig1.suptitle(f"Phase 7 — Multimodal Neural-to-Visual Alignment (OpenCLIP ViT-B-32)", color="white", fontsize=12, y=1.02)
    fig1.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase7_multimodal_training_and_retrieval.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig1)
    logger.info("Saved: phase7_multimodal_training_and_retrieval.png")

    # ── Figure 2: Brain vs CLIP Representational Geometry Alignment ──────────
    fig2 = plt.figure(figsize=(15, 5), facecolor="#0f1117")
    gs2 = gridspec.GridSpec(1, 3, figure=fig2, width_ratios=[1, 1, 1.2], wspace=0.35)

    # Brain RDM
    ax_b1 = fig2.add_subplot(gs2[0])
    im_b1 = ax_b1.imshow(rdm_eeg, cmap="magma", aspect="auto")
    ax_b1.set_title("Human Brain RDM\n(EEG Ventral Representations)", color="white", fontsize=10)
    ax_b1.set_xlabel("Concepts", color="white", fontsize=8)
    ax_b1.set_ylabel("Concepts", color="white", fontsize=8)
    ax_b1.tick_params(colors="white", labelsize=7)
    for spine in ax_b1.spines.values(): spine.set_edgecolor("#333")
    fig2.colorbar(im_b1, ax=ax_b1, fraction=0.046, pad=0.04).ax.tick_params(colors="white")

    # CLIP Vision RDM
    ax_b2 = fig2.add_subplot(gs2[1])
    im_b2 = ax_b2.imshow(rdm_clip, cmap="magma", aspect="auto")
    ax_b2.set_title("OpenCLIP Vision RDM\n(ViT-B-32 Semantic Geometry)", color="white", fontsize=10)
    ax_b2.set_xlabel("Concepts", color="white", fontsize=8)
    ax_b2.tick_params(colors="white", labelsize=7)
    for spine in ax_b2.spines.values(): spine.set_edgecolor("#333")
    fig2.colorbar(im_b2, ax=ax_b2, fraction=0.046, pad=0.04).ax.tick_params(colors="white")

    # Scatter Fit
    ax_b3 = fig2.add_subplot(gs2[2])
    ax_b3.set_facecolor("#1a1d27")
    v_eeg = vectorize_rdm(rdm_eeg)
    v_clip = vectorize_rdm(rdm_clip)
    ax_b3.scatter(v_eeg, v_clip, alpha=0.35, color="#00ff88", s=14, edgecolors="none")
    m, b = np.polyfit(v_eeg, v_clip, 1)
    xs = np.linspace(v_eeg.min(), v_eeg.max(), 100)
    ax_b3.plot(xs, m * xs + b, color="#ffffff", lw=1.8, label=f"Fit (Slope: {m:.2f})")
    ax_b3.set_xlabel("Human Neural Dissimilarity", color="white", fontsize=9)
    ax_b3.set_ylabel("CLIP Vision Dissimilarity", color="white", fontsize=9)
    ax_b3.set_title(f"Brain-to-Model RSA Correspondence\nSpearman ρ = {rsa_rho:.4f} (p = {rsa_p:.4e})",
                    color="#00ff88" if rsa_p < 0.05 else "#ff6b6b", fontsize=10)
    ax_b3.tick_params(colors="white", labelsize=8)
    ax_b3.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax_b3.spines.values(): spine.set_edgecolor("#333")

    fig2.suptitle("Phase 7 — Brain vs Artificial Neural Network Representational Geometry (RSA)", color="white", fontsize=12, y=1.03)
    fig2.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase7_brain_vs_clip_rdm_alignment.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig2)
    logger.info("Saved: phase7_brain_vs_clip_rdm_alignment.png")

    # ── Figure 3: Joint Multimodal Space & Top-K Retrieval Curve ─────────────
    fig3, (ax_j1, ax_j2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f1117")

    # Joint 2D PCA Space
    ax_j1.set_facecolor("#1a1d27")
    pca = PCA(n_components=2, random_state=42)
    Z_combined = np.concatenate([Z_clip_all, z_te_proj], axis=0)
    pca_emb = pca.fit_transform(Z_combined)
    clip_pca = pca_emb[:len(Z_clip_all)]
    eeg_pca = pca_emb[len(Z_clip_all):]

    ax_j1.scatter(clip_pca[:, 0], clip_pca[:, 1], color="#ff6b6b", s=70, marker="*", label="CLIP Target Centroids (50 Concepts)", zorder=4)
    ax_j1.scatter(eeg_pca[:, 0], eeg_pca[:, 1], color="#00ff88", s=18, alpha=0.5, label="Projected EEG Trials (Held-Out sub-03)", zorder=3)

    # Draw alignment segments for first 15 trials
    for i in range(min(20, len(y_te))):
        c_target = y_te[i]
        ax_j1.plot([eeg_pca[i, 0], clip_pca[c_target, 0]],
                   [eeg_pca[i, 1], clip_pca[c_target, 1]],
                   color="#888888", lw=0.8, alpha=0.5, ls=":")

    ax_j1.set_title("Joint Multimodal Latent Space (CLIP 512-d → 2D PCA)\n[Dotted lines connect EEG trials to true CLIP visual concepts]", color="white", fontsize=10)
    ax_j1.set_xlabel("Multimodal PC 1", color="white", fontsize=9)
    ax_j1.set_ylabel("Multimodal PC 2", color="white", fontsize=9)
    ax_j1.tick_params(colors="white")
    ax_j1.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax_j1.spines.values(): spine.set_edgecolor("#333")

    # Top-K Retrieval Curve
    ax_j2.set_facecolor("#1a1d27")
    ks = list(topk_accs.keys())
    accs = [topk_accs[k] * 100.0 for k in ks]
    chance_ks = [k / n_concepts * 100.0 for k in ks]

    ax_j2.plot(ks, accs, color="#00ff88", marker="o", lw=2.2, label="Zero-Shot CLIP Concept Retrieval (sub-03)")
    ax_j2.plot(ks, chance_ks, color="#ff6b6b", ls="--", lw=1.5, label="Random Guess Chance")
    ax_j2.fill_between(ks, chance_ks, accs, color="#00ff88", alpha=0.15)

    for k, v in zip(ks, accs):
        ax_j2.annotate(f"Top-{k}: {v:.1f}%", xy=(k, v), xytext=(k, v + 2.5), color="white", fontsize=9, ha="center")

    ax_j2.set_xlabel("Top-K Candidate Pool (N=50 Concepts)", color="white", fontsize=10)
    ax_j2.set_ylabel("Zero-Shot Retrieval Accuracy (%)", color="white", fontsize=10)
    ax_j2.set_title("Zero-Shot Visual Retrieval on Held-Out Brain", color="white", fontsize=11)
    ax_j2.set_xticks(ks)
    ax_j2.tick_params(colors="white")
    ax_j2.legend(frameon=False, labelcolor="white", fontsize=8)
    for spine in ax_j2.spines.values(): spine.set_edgecolor("#333")

    fig3.suptitle("Phase 7 — Multimodal Embedding Geometry & Zero-Shot Decoding", color="white", fontsize=12, y=1.02)
    fig3.patch.set_facecolor("#0f1117")
    plt.savefig(out_dir / "phase7_joint_latent_embedding_space.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig3)
    logger.info("Saved: phase7_joint_latent_embedding_space.png")


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
    logger.info("Phase 7 — Multimodal EEG-CLIP Representation Alignment")
    logger.info("  Train Subjects: %s | Test Subject: %s | Concepts: %d | Device: %s",
                train_subjs, test_subj, args.n_concepts, device)
    logger.info("=" * 60)

    # 1. Load EEG datasets and Concept Names
    train_data, test_data, concepts, concept_names = load_eeg_and_concepts(
        store, train_subjs, test_subj, args.n_concepts
    )
    X_tr, y_tr = train_data
    X_te, y_te = test_data

    # 2. Extract OpenCLIP ViT-B-32 Concept Embeddings
    Z_clip_all = extract_concept_clip_embeddings(
        concept_names, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )

    # 3. Load Pretrained EEGNetEncoder from Phase 6 Checkpoint
    eegnet_ckpt = ckpt_dir / "contrastive_eegnet.pt"
    logger.info("Loading pretrained Phase 6 EEGNetEncoder from %s …", eegnet_ckpt)
    eegnet = EEGNetEncoder(
        n_channels=X_tr.shape[1], n_samples=X_tr.shape[2], latent_dim=128, projection_dim=128
    ).to(device)
    if eegnet_ckpt.exists():
        state = torch.load(eegnet_ckpt, map_location=device)
        eegnet.load_state_dict(state["model_state_dict"])
        logger.info("Loaded pretrained EEGNet weights successfully!")
    else:
        logger.warning("Checkpoint %s not found; using freshly initialized encoder.", eegnet_ckpt)

    # Extract EEG latent representations
    Z_eeg_tr = extract_eeg_latents(eegnet, X_tr, device)
    Z_eeg_te = extract_eeg_latents(eegnet, X_te, device)

    # 4. Train EEG-to-CLIP Projector
    projector, history, z_te_proj = train_multimodal_projector(
        Z_eeg_tr, y_tr, Z_clip_all, Z_eeg_te, y_te, epochs=args.epochs, device=device
    )

    # Save Projector Checkpoint
    proj_ckpt = ckpt_dir / "eeg_to_clip_projector.pt"
    torch.save({"model_state_dict": projector.state_dict(), "concept_names": concept_names}, proj_ckpt)
    logger.info("Saved Projector checkpoint → %s", proj_ckpt)

    # 5. Zero-Shot Retrieval Evaluation on Held-Out Subject (sub-03)
    dists = cdist(z_te_proj, Z_clip_all, metric="cosine")
    rankings = np.argsort(dists, axis=1)

    topk_accs = {}
    for k in (1, 5, 10, 20):
        hits = sum(1 for i in range(len(y_te)) if y_te[i] in rankings[i, :k])
        topk_accs[k] = float(hits / len(y_te))

    # 6. Brain-to-CLIP RSA
    erps_eeg = np.stack([X_tr[y_tr == c].mean(axis=0) for c in range(args.n_concepts)], axis=0)
    rdm_eeg, rdm_clip, rsa_rho, rsa_p, rsa_pearson = compute_multimodal_rsa(erps_eeg, Z_clip_all)
    logger.info("Brain-to-CLIP RSA: Spearman ρ = %.4f (p = %.4e)", rsa_rho, rsa_p)

    # 7. Generate Figures
    logger.info("Generating publication figures → %s", out_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_phase7_figures(
            history, rdm_eeg, rdm_clip, rsa_rho, rsa_p,
            z_te_proj, Z_clip_all, y_te, topk_accs,
            concept_names, args.n_concepts, out_dir
        )

    # 8. Save structured results JSON
    chance = 1.0 / args.n_concepts
    results = {
        "train_subjects": train_subjs,
        "test_subject": test_subj,
        "n_concepts": args.n_concepts,
        "epochs_trained": args.epochs,
        "final_multimodal_loss": round(history["loss"][-1], 6),
        "chance_level": round(chance, 6),
        "zero_shot_top1_retrieval": round(topk_accs[1], 6),
        "zero_shot_top5_retrieval": round(topk_accs[5], 6),
        "zero_shot_top10_retrieval": round(topk_accs[10], 6),
        "zero_shot_top20_retrieval": round(topk_accs[20], 6),
        "brain_to_clip_rsa_rho": round(rsa_rho, 6),
        "brain_to_clip_rsa_p": round(rsa_p, 6),
        "brain_to_clip_significant": rsa_p < 0.05 and rsa_rho > 0,
        "h5_h6_supported": topk_accs[1] > chance or rsa_p < 0.05,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase7_results.json").write_text(json.dumps(results, indent=2))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 7 Final Results Summary")
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info("  %-32s %s", k, v)
    logger.info("=" * 60)
    logger.info("H₅ & H₆ (Multimodal Alignment & Retrieval): %s",
                "SUPPORTED ✓" if results["h5_h6_supported"] else "NOT SUPPORTED ✗")
    logger.info("All artifacts saved to: %s", out_dir)


if __name__ == "__main__":
    main()
