# shared-neural-space

**Are neural representations interoperable across human brains?**

A rigorous computational neuroscience project investigating whether a shared
latent EEG representational geometry exists across individuals, moving from
population-scale ERP analysis through subject-invariant contrastive alignment
to zero-shot semantic retrieval and recognition-memory generalisation.

## Research Overview

Three datasets, three stages, eight pre-registered hypotheses (see [`HYPOTHESES.md`](HYPOTHESES.md)).

| Stage | Dataset | Question |
|-------|---------|----------|
| Perception | THINGS-EEG `ds003825` (50 subj, ~55 GB) | Is representational geometry shared? |
| Deep Representation | THINGS-EEG2 `nm000232` (10 subj, ~241 GB) | Does it transfer to unseen humans? |
| Imagery | `ds005815` (20 subj, ~10 GB) | Does imagery reinstate perception? |
| Memory | Essex `ds006142` (27 subj, ~24 GB) | Are recognition states shared? |

**Key design principle — ephemeral caching:**

```
Remote Raw → Temporary Recording → Processed Representation → Delete Raw
```

No dataset is downloaded in full. One recording is fetched at a time, processed,
compressed to embeddings/features, and the raw cache is deleted.

## Repository Structure

```
shared-neural-space/
├── configs/              # Per-dataset YAML configurations
├── src/
│   ├── data/             # EEGDash loaders + manifest builder
│   ├── preprocessing/    # Pipeline, montage harmonisation, cache manager
│   ├── features/         # Spectral, temporal, covariance features
│   ├── models/           # EEGNet, Conformer, contrastive, adversarial
│   ├── alignment/        # RSA, Procrustes, CCA, SRM
│   ├── evaluation/       # LOSO, retrieval metrics, zero-shot, probes
│   └── visualization/    # ERP, topomap, latent-space plots
├── scripts/              # CLI entry points
├── notebooks/            # Stage-by-stage interactive walkthroughs
├── tests/                # pytest suite
├── artifacts/            # Generated (gitignored): manifests, features, models
├── HYPOTHESES.md         # Pre-registered hypotheses — frozen before data inspection
├── requirements.txt
└── pyproject.toml
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Phase 0 — Remote data access proof

```bash
# Inspect one THINGS-EEG recording and immediately delete the raw cache
python scripts/preprocess_subject.py \
    --config configs/things_population.yaml \
    --subject sub-01 \
    --dry-run
```

Or open [`notebooks/01_remote_access.ipynb`](notebooks/01_remote_access.ipynb)
in Colab.

### 3. Run tests

```bash
pytest tests/ -v
```

## Storage Architecture

| Level | Content | Location | Lifecycle |
|-------|---------|----------|-----------|
| 0 | Raw EEG | OpenNeuro / NEMAR | Remote only |
| 1 | Processed epochs | Local / Colab `/content/cache` | Temporary |
| 2 | Features / embeddings | `artifacts/features/*.zarr` | Persistent |

## Important Notes

- **Channel heterogeneity:** `ds003825` has 48 subjects with 63 channels and 2
  with 128 channels. The MVP uses only the 48 homogeneous subjects. A
  `ChannelHarmonizer` is available for later common-electrode projection.
- **Recording-level caching:** THINGS-EEG2 has 638 recordings / 10 subjects.
  The cache manager operates at the **recording level**, not subject level.
- **Leakage-free preprocessing:** All normalization, feature, and alignment
  parameters are fit exclusively on training subjects.
- **UMAP is visualization only.** Quantitative claims rest on RSA, retrieval
  metrics, and permutation tests.

## License

Code: MIT. Datasets are subject to their respective OpenNeuro/NEMAR licences
(THINGS-EEG: CC0; others: see individual dataset pages).
