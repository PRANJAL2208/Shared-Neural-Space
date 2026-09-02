# Frozen Scientific Hypotheses

> **Status: PRE-REGISTERED — Do not revise after data inspection.**
>
> All eight hypotheses below are frozen before any data analysis begins.
> Revisions to analysis parameters (time windows, bandpass frequencies, etc.)
> must be documented explicitly as *exploratory* deviations and reported separately.

---

## Central Research Question

> **Are neural representations interoperable across human brains?**
>
> Specifically: does there exist a computationally recoverable shared coordinate
> system in which EEG activity from different individuals can be aligned, such
> that stimulus-related information generalises across people?

---

## Hypotheses

### H1 — Cross-Subject ERP Similarity (Checkpoint A)

**Claim:** Average EEG responses to the same semantic concept are more similar
across different individuals than responses to different concepts.

$$S_{\text{same}} = \text{sim}(\overline{X}_{A,c},\, \overline{X}_{B,c})
\quad > \quad
S_{\text{different}} = \text{sim}(\overline{X}_{A,c},\, \overline{X}_{B,d}),
\quad c \neq d$$

**Test:** Paired permutation test on $\Delta S = S_{\text{same}} - S_{\text{different}}$.
Cluster-based correction across time samples.

**Dataset:** THINGS-EEG `ds003825` (48 subjects, 63-channel montage).

---

### H2 — Shared Representational Geometry (Checkpoint B)

**Claim:** Pairwise concept-level dissimilarity matrices (RDMs) computed
independently per subject are positively correlated.

$$\rho\!\left(\text{RDM}_i,\, \text{RDM}_j\right) > 0$$

**Test:** Spearman rank correlation on vectorised upper triangles of RDMs.
Significance via label-permutation null distribution.

**Dataset:** THINGS-EEG `ds003825`.

---

### H3 — Cross-Subject Stimulus Generalisation (Checkpoint C)

**Claim:** A decoder trained on data from subjects $\{1, \ldots, n-1\}$ predicts
stimulus category for the completely held-out subject $n$ significantly above
permutation chance.

**Test:** Nested Leave-One-Subject-Out cross-validation. Permutation test on
held-out subject accuracy.

**Dataset:** THINGS-EEG2 `nm000232`.

---

### H4 — Latent Subject Invariance (Checkpoint C)

**Claim:** After subject-adversarial contrastive alignment, a linear subject-
identification probe trained on embeddings $Z$ performs at chance, while a
concept probe trained on the same $Z$ performs significantly above chance.

$$\text{Accuracy}_{\text{concept}} \uparrow, \quad \text{Accuracy}_{\text{subject}} \downarrow \text{ (toward chance)}$$

**Test:** Train logistic regression probes on frozen $Z$ before and after DANN
training. Report accuracy with 95 % confidence intervals.

**Dataset:** THINGS-EEG2 `nm000232`.

---

### H5 — Semantic Neural Alignment

**Claim:** EEG embeddings are more similar to image embeddings depicting the
same concept than to embeddings of different concepts.

$$\text{sim}(Z_{\text{EEG}},\, Z_{\text{image}}^{\text{correct}})
> \text{sim}(Z_{\text{EEG}},\, Z_{\text{image}}^{\text{incorrect}})$$

**Test:** Contrastive retrieval: Top-1 accuracy, Top-5, MRR, Recall@{1,5,10}.
**Dataset:** THINGS-EEG2 `nm000232` + CLIP/DINOv2 image embeddings.

---

### H6 — Zero-Shot Semantic Generalisation

**Claim:** The neural representation retrieves correct semantic information for
concepts that were entirely absent from training (all trials, all images).

**Test:** Multi-tier holdout:
1. Trial holdout (easiest)
2. Image holdout
3. Concept holdout (all trials for concept removed from train)
4. Subject holdout
5. Subject + concept holdout (hardest)

**Null:** chance-level retrieval accuracy under permutation.

**Dataset:** THINGS-EEG2 `nm000232`.

---

### H7 — Perception / Imagery Reinstatement

**Claim:** Within `ds005815`, the representational similarity between perception
and imagery is higher for matched conditions than mismatched conditions.

$$\text{sim}(Z_{\text{perception, cond } c},\, Z_{\text{imagery, cond } c})
> \text{sim}(Z_{\text{perception, cond } c},\, Z_{\text{imagery, cond } d}),
\quad c \neq d$$

**Note:** This is analysed *within* `ds005815` first. Cross-dataset alignment
(THINGS embeddings → imagery space) is treated as *exploratory*.

**Dataset:** Imagery `ds005815`.

---

### H8 — Recognition-State Generalisation

**Claim:** A classifier trained to distinguish `NOT_RECOGNISED`,
`RECOGNISED_NOT_REMEMBERED`, and `RECOGNISED_AND_REMEMBERED` on a subset of
subjects generalises to completely unseen subjects significantly above chance.

**Secondary test:** Align EEG around recognition trigger time ($-2\text{s}$ to
$+2\text{s}$) and ask whether a common latent transition precedes conscious
recognition.

**Dataset:** Essex Movie Memory `ds006142`.

---

## Experimental Ladder

Do not skip stages. Each stage proceeds only if the previous one is resolved.

| Stage | Analysis | Hypothesis |
|------:|---------|-----------|
| 0 | Remote data access & cache teardown | — |
| 1 | Metadata/event validity | — |
| 2 | Signal preprocessing validation | — |
| 3 | Same-vs-different ERP (MVP-A broad categories) | H1 |
| 4 | Time-resolved cross-subject similarity | H1 |
| 5 | Spectral band features | exploratory |
| 6 | Representational Similarity Analysis | H2 |
| 7 | CSP / LDA / SVM baselines | H3 |
| 8 | Nested LOSO classification | H3 |
| 9 | EEGNet deep baseline | H3 |
| 10 | EEG Conformer | H3 |
| 11 | Hierarchical contrastive EEG↔EEG alignment | H4 |
| 12 | DANN subject-adversarial invariance + probes | H4 |
| 13 | EEG ↔ image (CLIP/DINOv2) alignment | H5 |
| 14 | Concept zero-shot retrieval | H6 |
| 15 | Subject zero-shot retrieval | H6 |
| 16 | Subject + concept zero-shot | H6 |
| 17 | Perception ↔ imagery reinstatement | H7 |
| 18 | Recognition-moment analysis & generalisation | H8 |

---

## MVP Success Criterion (Milestone #1)

Before inspecting any results, define:

$$H_0: \Delta S = 0 \quad \text{vs} \quad H_1: \Delta S > 0$$

where $\Delta S = S_{\text{same}} - S_{\text{different}}$.

**Acceptance threshold:** $p < 0.05$ (permutation), $\Delta S > 0$ across
majority of concept pairs, at least one temporal cluster surviving correction.

**If $\Delta S \leq 0$:** Investigate preprocessing, electrode alignment, and
time-window choice before advancing to deep learning.

---

*Last frozen: 2026-09-01. Author: shared-neural-space project.*
