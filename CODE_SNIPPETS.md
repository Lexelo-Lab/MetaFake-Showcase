# MetaFake — Selected Code Snippets

> These are **abridged, illustrative excerpts** to show the core ideas and how
> they are implemented. Boilerplate, the data pipeline, the full training loop,
> and several implementation details are intentionally omitted. The complete,
> runnable source and trained weights are maintained privately.

---

## 1. The model: three experts, one fusion head

The top-level model wraps a **frozen** LaDeDa and adds the two new experts. The
forward pass produces five scalar signals and fuses them into one score.

```python
class MetaFake(nn.Module):
    def __init__(self, base_ckpt, use_dct=True, use_clip=True):
        super().__init__()
        self.ladeda = LaDeDaBackbone(base_ckpt)   # frozen, loads pretrained weights
        self.dct    = DCTExpert()   if use_dct  else None   # new, trainable
        self.clip   = CLIPExpert()  if use_clip else None   # frozen backbone + small head
        self.fusion = FusionHead(n_inputs=5)                # tiny MLP

    def forward_with_aux(self, x):
        # Expert 1: frozen LaDeDa -> image logit + patch-map statistics
        laD_logit, laD_std, laD_max = self.ladeda(x)
        # Expert 2: frequency-domain logit
        dct_logit  = self.dct(x)  if self.dct  else torch.zeros_like(laD_logit)
        # Expert 3: CLIP semantic logit
        clip_logit = self.clip(x) if self.clip else torch.zeros_like(laD_logit)

        v = torch.cat([laD_logit, laD_std, laD_max, dct_logit, clip_logit], dim=1)
        return {"fusion": self.fusion(v),
                "dct": dct_logit, "clip": clip_logit, "ladeda": laD_logit}
```

The base detector is fully frozen, so its weights never change during training:

```python
for p in self.ladeda.parameters():
    p.requires_grad = False
self.ladeda.eval()   # also keeps BatchNorm statistics fixed
```

---

## 2. The key trick: no-regression fusion initialization

The fusion head is initialized so that, **before any training**, its output is a
monotone function of the LaDeDa logit alone. Since AP and AUC depend only on the
*ranking* of scores, MetaFake reproduces the base detector's ranking exactly at
step zero — the upgrade can never start below the baseline.

```python
class FusionHead(nn.Module):
    def __init__(self, n_inputs=5, hidden=16, ladeda_logit_idx=0):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.weight[:, ladeda_logit_idx] = 1.0   # pass through LaDeDa only
            self.fc1.bias.zero_()
            self.fc2.weight.fill_(1.0 / hidden)          # uniform positive average
            self.fc2.bias.zero_()
    # forward: fc2(GELU(fc1(v)))  ->  starts == LaDeDa, then learns corrections
```

This is what makes the upgrade safe to deploy: the new experts can only move the
score away from the LaDeDa baseline if doing so lowers the loss.

---

## 3. A clever detail: block-DCT as a fixed convolution

The frequency expert needs a per-block DCT. Instead of using an FFT (which is not
well supported on all hardware backends), we precompute the cosine basis once and
run it as a **fixed, non-trainable convolution** with stride 16. This is exactly
a block DCT, but it runs anywhere — including Apple Silicon (MPS).

```python
# luma -> non-overlapping 16x16 block DCT, implemented as one strided conv
coeffs = F.conv2d(luma, self.dct_kernel, stride=16)   # dct_kernel: fixed cosine basis
feats  = torch.log1p(coeffs.abs())                    # log-magnitude, sign-invariant
logit  = self.cnn(feats)                              # small 3-layer CNN -> 1 logit
```

*(The construction of `dct_kernel` from the DCT-II cosine basis is omitted here.)*

---

## 4. Training objective: keep every expert useful

We train only the new parts. Besides the main loss on the fused score, each new
expert gets a small **auxiliary loss** on its own logit, so the fusion cannot
simply ignore the new experts and rely on the (initially dominant) LaDeDa signal.

```python
main_loss = bce(out["fusion"], y)
aux_loss  = bce(out["dct"], y) + bce(out["clip"], y)
loss = main_loss + 0.1 * aux_loss          # lambda = 0.1
loss.backward()                            # gradients flow only to trainable params
```

Trainable surface: **~287K of ~318M parameters (0.09%)**. Everything else (the
LaDeDa backbone and the CLIP encoder) is frozen.

---

## 5. Real-world augmentation (concept)

To survive social-media processing, training images pass through a SAFE-style
pipeline that imitates how platforms degrade images:

```python
# applied randomly during training (abridged)
img = random_downscale_then_upscale(img, ratio=0.5..1.0)   # random interpolation
img = double_jpeg(img, quality=70..95)                     # re-compression
img = brightness_contrast_jitter(img)
```

This teaches the model to ignore resizing and compression and focus on the
actual synthesis signal.

---

*These excerpts illustrate the ideas. The complete implementation, integration
glue, evaluation tools, and trained checkpoints are not included in this
showcase.*
