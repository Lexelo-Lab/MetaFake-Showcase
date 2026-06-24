# MetaFake — Project Showcase

**Real-Time Deepfake Detection for the Diffusion-Transformer Era**

Erik Paperniuk · Viacheslav Kartushin · Naeem Ul Islam
International Bachelor Program in Informatics, Yuan Ze University

> This folder is a high-level showcase of our ideas and results. It contains the
> concept, architecture, and measured outcomes. The full source code, trained
> model weights, and training pipeline live in our private project and are not
> included here.

---

## 1. The Problem

AI image generators improved dramatically in 2024–2026. A detector that worked
well in 2024 quietly loses accuracy as new generators appear, because each new
generator architecture leaves different traces.

The base detector we build on, **LaDeDa (2024)**, works by scanning tiny
9×9 patches and looking for **upsampling artifacts** left by GAN and U-Net
diffusion decoders. It is fast and accurate, reaching 93.7% mAP on real-world
social-media images.

The problem: the newest generators broke this assumption.

| Generator family | Example models | Why LaDeDa struggles |
|---|---|---|
| Diffusion transformers + rectified flow | FLUX.1, Stable Diffusion 3 / 3.5 | No transpose-conv upsampling → weak artifacts |
| Autoregressive / multimodal | GPT-4o image gen, Nano Banana | Generated token-by-token, no local fingerprint |
| Distilled few-step | SDXL-Turbo, FLUX-schnell | Different artifacts than many-step models |

Retraining a large detector from scratch every few months is expensive. We
wanted a cheaper, repeatable way to keep a deployed detector current.

---

## 2. Our Idea

**Keep the existing detector frozen. Stack a small fusion layer on top that adds
two new "experts," each targeting a generator family the base detector misses.**

```
                                    ┌─────────────────────────────────────┐
   Image ──► LaDeDa (FROZEN)  ────► │ image logit, patch spread, patch max │ ─┐
         ──► DCT frequency expert ─► │ frequency logit                      │ ─┼─► Fusion ─► Score
         ──► CLIP vision (FROZEN) ─► │ semantic logit                       │ ─┘   (tiny MLP)
                  + small head       └─────────────────────────────────────┘
```

Three experts, each chosen on purpose:

1. **LaDeDa (frozen)** — keeps the original strength: upsampling artifacts.
2. **Frequency expert** — looks at the image in the frequency domain (block DCT).
   Rectified-flow models hide their spatial artifacts but still leave statistical
   biases in the frequencies.
3. **Semantic expert** — a frozen CLIP vision model. High-quality closed models
   (GPT-4o, Nano Banana) leave no local artifact, but their global texture and
   content statistics still differ slightly from real photos, and CLIP captures
   this.

A tiny fusion network combines five numbers into one final score.

### Key innovations

- **No-regression by design.** The fusion layer is initialized to copy the base
  detector exactly. Before any training, our model produces the same ranking as
  LaDeDa, so the upgrade can never start below the baseline. The new experts only
  learn to *correct* the base detector where it is wrong.
- **Tiny trainable footprint.** Only ~287K of ~318M total parameters are trained
  (0.09%). Everything heavy stays frozen.
- **Trains on a laptop.** Full training finishes in **under one hour on an Apple
  M1 Max**, no datacenter GPU required.
- **Real-time friendly.** Inference stays within a small factor of the original
  LaDeDa latency; the semantic expert runs once per image and can be switched off
  for maximum speed.

---

## 3. Results

All comparisons below keep the LaDeDa weights **identical** — only the small
fusion layer and the two new experts are trained.

### WildRF — real-world social media (Reddit / Twitter / Facebook)

| Platform | LaDeDa AP | MetaFake AP | Δ AP | LaDeDa Acc | MetaFake Acc | Δ Acc |
|---|---|---|---|---|---|---|
| Reddit | 96.0 | **98.2** | +2.2 | 91.8 | **96.4** | +4.6 |
| Twitter | 92.1 | **94.6** | +2.5 | 82.0 | **91.9** | +10.0 |
| Facebook | 92.6 | **93.7** | +1.2 | 81.9 | **91.9** | +10.0 |
| **Mean** | **93.5** | **95.5** | **+2.0** | 85.2 | **93.4** | **+8.2** |

Mean average precision up **+2.0 points**, accuracy up **+10 points** on the two
hardest platforms — without touching the base detector.

### Synthbuster — nine classic generators

| Model | Acc | AP | AUC | Precision | Recall |
|---|---|---|---|---|---|
| LaDeDa | 95.3 | 99.54 | 98.64 | 99.77 | 95.00 |
| **MetaFake** | **98.9** | **99.83** | **99.13** | 99.12 | **99.67** |

Here both models rank perfectly (per-generator AP = 1.0), so the gain is in
**confidence**: the base detector misses 50 of 900 fakes near the threshold;
ours misses only 3 (recall 95.0% → 99.7%).

### Figures

See the [`figures/`](figures/) folder:

- `roc_curve.png`, `pr_curve.png` — ROC and precision-recall curves
- `cm_ladeda.png`, `cm_metafake.png` — confusion matrices (MetaFake catches all
  fakes; trades a little real-image precision for it)
- `metrics_bar.png` — side-by-side metric summary

---

## 4. What Each Benchmark Shows

- **Synthbuster (old generators):** the base detector already ranks perfectly;
  our experts make it *more confident*, improving accuracy and recall.
- **WildRF (real-world):** harder and messier; our experts improve the actual
  *ranking quality* (AP) and give large accuracy gains where the base was weak.
- **Modern generators (FLUX, SD3.5, GPT-Image, Nano Banana, Midjourney v6,
  Ideogram 3):** this is where the base detector is blind and where we expect the
  new experts to matter most. We assembled a dedicated test slice from a public
  dataset; evaluation here is ongoing.

---

## 5. Summary

We show that a deployed, real-time patch-based deepfake detector does **not** need
to be retrained or replaced to handle the newest generators. A small fusion layer
with a frequency expert and a semantic expert, trained in under an hour on a
laptop, improves real-world performance with zero risk of regression. The three
experts map cleanly onto the three modern generator families: upsampling-based,
rectified-flow, and autoregressive.

*Source code, model weights, and the full training pipeline are maintained
privately and are not part of this showcase.*
