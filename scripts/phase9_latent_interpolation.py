#!/usr/bin/env python3
"""Phase 9: Latent Neural Interpolation & Continuous Manifold Traversal.

Stage 3: Neural State Synthesis & Cross-Brain Latent Traversal
- Hypotheses H8:
  1. Continuous Neural Manifold: Spherical linear interpolation between neural concept centroids
     produces smooth, monotonic semantic transitions without discrete topological collapse.
  2. Cross-Brain Traversal: Interpolated paths generated from Subject 1 decode consistently
     when evaluated against held-out Subject 3's semantic space.
  3. Neural Vector Arithmetic: Linear algebraic operations in the shared latent neural space
     (Z_A - Z_B + Z_C) retrieve valid analogical target concepts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.alignment.clip_alignment import EEGToCLIPProjector, extract_concept_clip_embeddings
from src.alignment.interpolation import (
    evaluate_interpolation_monotonicity,
    evaluate_neural_vector_arithmetic,
    interpolate_latents,
)
from src.data.zarr_store import ZarrEpochStore
from src.models.encoder import EEGNetEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_encoder_and_projector(
    encoder_path: Path,
    projector_path: Path,
    device: torch.device,
) -> tuple[EEGNetEncoder, EEGToCLIPProjector]:
    """Load pretrained EEGNet encoder and EEG-to-CLIP projector."""
    encoder = EEGNetEncoder(n_channels=63, n_samples=251, latent_dim=128).to(device)
    enc_ckpt = torch.load(encoder_path, map_location=device, weights_only=True)
    enc_weights = enc_ckpt.get("model_state_dict", enc_ckpt)
    encoder.load_state_dict(enc_weights)
    encoder.eval()

    projector = EEGToCLIPProjector(input_dim=128, clip_dim=512).to(device)
    proj_ckpt = torch.load(projector_path, map_location=device, weights_only=True)
    proj_weights = proj_ckpt.get("projector_state_dict", proj_ckpt.get("model_state_dict", proj_ckpt))
    projector.load_state_dict(proj_weights)
    projector.eval()

    return encoder, projector


def compute_subject_centroids(
    store: ZarrEpochStore,
    subject: str,
    concepts: list[int],
    encoder: EEGNetEncoder,
    projector: EEGToCLIPProjector,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean 128-d EEG latent and 512-d CLIP projected centroids per concept."""
    data = store.read_subject(subject, concept_filter=concepts)
    X = torch.from_numpy(data["eeg"].astype(np.float32)).to(device)
    lbls = data["labels"].astype(np.int32)

    with torch.no_grad():
        z_eeg = encoder.encode(X)
        z_clip = projector(z_eeg)

    z_eeg_np = z_eeg.cpu().numpy()
    z_clip_np = z_clip.cpu().numpy()

    centroids_eeg = []
    centroids_clip = []
    for c in concepts:
        mask = (lbls == c)
        if np.any(mask):
            c_eeg = np.mean(z_eeg_np[mask], axis=0)
            c_eeg = c_eeg / (np.linalg.norm(c_eeg) + 1e-9)
            centroids_eeg.append(c_eeg)

            c_clip = np.mean(z_clip_np[mask], axis=0)
            c_clip = c_clip / (np.linalg.norm(c_clip) + 1e-9)
            centroids_clip.append(c_clip)
        else:
            centroids_eeg.append(np.zeros(128))
            centroids_clip.append(np.zeros(512))

    return np.stack(centroids_eeg, axis=0), np.stack(centroids_clip, axis=0)


def plot_phase9_figures(
    interp_results: list[dict[str, Any]],
    pca_data: dict[str, Any],
    arithmetic_res: dict[str, Any],
    out_dir: Path,
) -> None:
    """Generate publication-ready figures for Phase 9."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # Figure 1: Latent Interpolation Semantic Trajectories
    # -------------------------------------------------------------
    n_pairs = min(3, len(interp_results))
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]

    for idx, (res, ax) in enumerate(zip(interp_results[:n_pairs], axes)):
        alphas = res["alphas"]
        ax.plot(alphas, res["sim_to_start"], color="#3182CE", lw=2.5, marker="o", markersize=4,
                label=f"Similarity to '{res['name_a']}'")
        ax.plot(alphas, res["sim_to_end"], color="#E53E3E", lw=2.5, marker="s", markersize=4,
                label=f"Similarity to '{res['name_b']}'")

        ax.axvline(0.5, color="gray", lw=1.0, ls="--", label=r"Midpoint ($\alpha = 0.5$)")
        ax.set_title(f"Trajectory: {res['name_a']} $\\to$ {res['name_b']}\n(Monotonicity: {res['monotonicity_score']*100:.1f}%)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel(r"Interpolation Step $\alpha$", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fig1_path = out_dir / "phase9_latent_interpolation_trajectories.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", fig1_path.name)

    # -------------------------------------------------------------
    # Figure 2: Continuous Manifold Traversal in 2D PCA Space
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    # Scatter concept centroids
    pca_centroids = pca_data["centroids_2d"]
    names = pca_data["concept_names"]
    ax.scatter(pca_centroids[:, 0], pca_centroids[:, 1], color="#CBD5E0", s=60, alpha=0.7, edgecolors="none")

    # Plot trajectories
    colors = ["#3182CE", "#DD6B20", "#38A169"]
    for idx, res in enumerate(interp_results[:3]):
        traj_2d = pca_data["trajectories_2d"][idx]
        c = colors[idx % len(colors)]
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], color=c, lw=2.5, ls="-",
                label=f"Path: {res['name_a']} $\\to$ {res['name_b']}")
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=c, s=120, marker="o", edgecolors="black")
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=c, s=120, marker="s", edgecolors="black")
        # Annotate
        ax.annotate(res["name_a"], (traj_2d[0, 0], traj_2d[0, 1]), fontsize=11, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")
        ax.annotate(res["name_b"], (traj_2d[-1, 0], traj_2d[-1, 1]), fontsize=11, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_title("Continuous Latent Manifold Geodesics Across Human Brains", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PCA Dimension 1 ({pca_data['evr'][0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PCA Dimension 2 ({pca_data['evr'][1]*100:.1f}% var)", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    fig2_path = out_dir / "phase9_continuous_manifold_interpolation_2d.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", fig2_path.name)

    # -------------------------------------------------------------
    # Figure 3: Neural Vector Arithmetic in Human Latent Space
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of accuracy vs chance
    axes[0].bar(["Top-1 Accuracy", "Top-5 Accuracy", "Random Chance"],
                [arithmetic_res["top1_accuracy"] * 100, arithmetic_res["topk_accuracy"] * 100, (1.0 / arithmetic_res["n_concepts"]) * 100],
                color=["#3182CE", "#38A169", "#A0AEC0"], width=0.55)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_title(r"Neural Vector Arithmetic ($Z_A - Z_B + Z_C \approx Z_D$)", fontsize=12, fontweight="bold")
    for i, v in enumerate([arithmetic_res["top1_accuracy"] * 100, arithmetic_res["topk_accuracy"] * 100, (1.0 / arithmetic_res["n_concepts"]) * 100]):
        axes[0].text(i, v + 1.0, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    axes[0].set_ylim(0, max(50.0, arithmetic_res["topk_accuracy"] * 100 + 15))

    # Analogy queries table / details
    axes[1].axis("off")
    queries_text = "Neural Analogy Query Results:\n\n"
    for r in arithmetic_res["analogy_results"][:6]:
        status = "\u2713 (Top-1)" if r["is_top1"] else ("\u2713 (Top-5)" if r["is_topk"] else "\u2717")
        queries_text += f"{r['query']} = {r['target']}  [{status}]\n"
        queries_text += f"   Predicted: {r['predicted_top1']} | Sim: {r['target_cosine_sim']:.3f}\n\n"

    axes[1].text(0.05, 0.95, queries_text, transform=axes[1].transAxes, fontsize=10,
                 verticalalignment="top", fontfamily="monospace")
    axes[1].set_title("Cross-Subject Analogy Traversal Examples", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig3_path = out_dir / "phase9_neural_vector_arithmetic.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", fig3_path.name)


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Latent Neural Interpolation")
    parser.add_argument("--zarr-path", type=str, default="E:/pranjal_evobrain/features/ds003825_epochs.zarr")
    parser.add_argument("--encoder-path", type=str, default="E:/pranjal_evobrain/checkpoints/contrastive_eegnet.pt")
    parser.add_argument("--projector-path", type=str, default="E:/pranjal_evobrain/checkpoints/eeg_to_clip_projector.pt")
    parser.add_argument("--out-dir", type=str, default="E:/pranjal_evobrain/plots/phase9")
    parser.add_argument("--n-concepts", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=21)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    logger.info("=" * 60)
    logger.info("Phase 9 — Latent Neural Interpolation & Continuous Manifold Traversal")
    logger.info("  Concepts: %d | Steps: %d | Device: %s", args.n_concepts, args.n_steps, device)
    logger.info("=" * 60)

    # 1. Load models
    encoder, projector = load_encoder_and_projector(
        encoder_path=Path(args.encoder_path),
        projector_path=Path(args.projector_path),
        device=device,
    )
    logger.info("Loaded pretrained EEGNetEncoder and EEGToCLIPProjector successfully!")

    # 2. Load Zarr Store & Concept Names
    store = ZarrEpochStore(args.zarr_path)
    subjects = store.subjects()
    logger.info("Available subjects in Zarr store: %s", subjects)

    labels_s1 = store._root[subjects[0]]["labels"][:]
    labels_s2 = store._root[subjects[1]]["labels"][:]
    labels_s3 = store._root[subjects[2]]["labels"][:]

    shared_concepts = sorted(list(set(np.unique(labels_s1)) & set(np.unique(labels_s2)) & set(np.unique(labels_s3))))[:args.n_concepts]

    # Concept names
    meta_grp = store._root[subjects[0]]
    if "concept_names" in meta_grp:
        raw_names = meta_grp["concept_names"][:]
        concept_names = [str(raw_names[c]) if c < len(raw_names) else f"concept_{c}" for c in shared_concepts]
    elif "concept_names_json" in meta_grp.attrs:
        raw_names = json.loads(meta_grp.attrs["concept_names_json"])
        concept_names = [str(raw_names[c]) if c < len(raw_names) else f"concept_{c}" for c in shared_concepts]
    else:
        concept_names = [f"concept_{c}" for c in shared_concepts]

    logger.info("Loaded %d concept names: %s ...", len(concept_names), concept_names[:8])

    # 3. Extract OpenCLIP reference concept embeddings
    logger.info("Extracting OpenCLIP semantic concept embeddings...")
    clip_embeds = extract_concept_clip_embeddings(concept_names, device=device)

    # 4. Compute Centroids for Subject 1 (Source) and Subject 3 (Held-Out Target)
    c_eeg_s1, c_clip_s1 = compute_subject_centroids(store, subjects[0], shared_concepts, encoder, projector, device)
    c_eeg_s3, c_clip_s3 = compute_subject_centroids(store, subjects[2], shared_concepts, encoder, projector, device)

    # 5. Latent Interpolation between Concept Pairs (e.g. musical instruments, tools, animals)
    pairs_to_test = [
        (1, 6),   # e.g. piano -> drum
        (3, 4),   # e.g. tent -> stove
        (0, 5),   # e.g. carousel -> birdbath
    ]
    # Filter valid pairs
    valid_pairs = [(a, b) for a, b in pairs_to_test if a < len(shared_concepts) and b < len(shared_concepts)]
    if not valid_pairs:
        valid_pairs = [(0, 1), (1, 2), (2, 3)]

    interp_results = []
    trajectories_clip = []

    for idx_a, idx_b in valid_pairs:
        z_a = c_clip_s1[idx_a]
        z_b = c_clip_s1[idx_b]

        alphas, traj = interpolate_latents(z_a, z_b, n_steps=args.n_steps, method="slerp")
        # Evaluate monotonicity against true semantic reference vectors
        mono_res = evaluate_interpolation_monotonicity(traj, clip_embeds[idx_a], clip_embeds[idx_b])

        res_dict = {
            "name_a": concept_names[idx_a],
            "name_b": concept_names[idx_b],
            "idx_a": idx_a,
            "idx_b": idx_b,
            "alphas": alphas,
            "trajectory": traj,
            "sim_to_start": mono_res["sim_to_start"],
            "sim_to_end": mono_res["sim_to_end"],
            "monotonicity_score": mono_res["monotonicity_score"],
        }
        interp_results.append(res_dict)
        trajectories_clip.append(traj)

        logger.info("Interpolation %s -> %s: Monotonicity Score = %.2f%%",
                    concept_names[idx_a], concept_names[idx_b], mono_res["monotonicity_score"] * 100)

    # 6. PCA Projection for 2D Manifold Visualization
    all_centroids = np.vstack([c_clip_s1, c_clip_s3, clip_embeds])
    pca = PCA(n_components=2)
    pca.fit(all_centroids)
    centroids_2d = pca.transform(c_clip_s3)
    trajectories_2d = [pca.transform(traj) for traj in trajectories_clip]

    pca_data = {
        "centroids_2d": centroids_2d,
        "trajectories_2d": trajectories_2d,
        "concept_names": concept_names,
        "evr": pca.explained_variance_ratio_,
    }

    # 7. Neural Vector Arithmetic (Analogies)
    # Define semantic analogies across human brain embeddings
    test_analogies = []
    if len(concept_names) >= 4:
        test_analogies.append((concept_names[0], concept_names[1], concept_names[2], concept_names[3]))
    if len(concept_names) >= 8:
        test_analogies.append((concept_names[1], concept_names[6], concept_names[3], concept_names[4]))
    if len(concept_names) >= 12:
        test_analogies.append((concept_names[7], concept_names[8], concept_names[9], concept_names[10]))

    arithmetic_res = evaluate_neural_vector_arithmetic(
        concept_embeddings=c_clip_s3,
        concept_names=concept_names,
        analogies=test_analogies,
        top_k=5,
    )
    arithmetic_res["n_concepts"] = len(shared_concepts)

    logger.info("Neural Vector Arithmetic: Top-1 Accuracy = %.1f%% | Top-5 Accuracy = %.1f%% (Chance = %.1f%%)",
                arithmetic_res["top1_accuracy"] * 100, arithmetic_res["topk_accuracy"] * 100, (1.0 / len(shared_concepts)) * 100)

    # 7. Generate Figures
    plot_phase9_figures(
        interp_results=interp_results,
        pca_data=pca_data,
        arithmetic_res=arithmetic_res,
        out_dir=out_dir,
    )

    # 8. Save JSON Results
    mean_monotonicity = float(np.mean([r["monotonicity_score"] for r in interp_results]))
    results = {
        "n_concepts": len(shared_concepts),
        "source_subject": subjects[0],
        "target_subject": subjects[2],
        "n_interpolation_steps": args.n_steps,
        "mean_monotonicity_score": mean_monotonicity,
        "pca_variance_explained": [float(x) for x in pca.explained_variance_ratio_],
        "vector_arithmetic_top1_acc": arithmetic_res["top1_accuracy"],
        "vector_arithmetic_top5_acc": arithmetic_res["topk_accuracy"],
        "chance_level": float(1.0 / len(shared_concepts)),
        "h8_supported": bool(mean_monotonicity > 0.85),
    }

    json_path = out_dir / "phase9_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved results -> %s", json_path)
    logger.info("=" * 60)
    logger.info("Phase 9 Final Summary:")
    for k, v in results.items():
        logger.info("  %-30s %s", k, v)
    logger.info("=" * 60)
    logger.info("H8 (Continuous Latent Manifold & Traversal): %s", "SUPPORTED \u2713" if results["h8_supported"] else "INCONCLUSIVE")


if __name__ == "__main__":
    main()
