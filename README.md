# Diffusion as Memory

Modelling memory in an SNR landscape to convey 'forgetting' by an LLM. Text is encoded into compact latent representations (a semantic anchor `u` and a detail latent `v0`). A diffusion denoiser then learns to recover stored memories from noisy latents.


### Training Losses

| Loss | Formula | Purpose |
|------|---------|---------|
| `loss_nce` | InfoNCE(`u`, `upos`) | Contrastive loss, pull together semantically similar inputs |
| `loss_x` | CrossEntropy(DecoderX(`v0`), `x`) | Force `v0` to preserve full text detail |
| `loss_y` | CrossEntropy(DecoderY(`u`), `y`) | Force `u` to capture summary semantics |
| **total** | `λ_u · loss_nce + λ_x · loss_x + λ_y · loss_y` | Combined objective |

---

## Project Structure

```
Diffusion_as_Memory/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── sbatch_scripts/               # SLURM job submission scripts
│
├── testing_refactored.py         # Main start point
│
├── data_loader/
│   └── data_loading.py           # MSRDataset, dummy dataloader
│
├── dataset_prep/                 # Data preparation and preprocessing scripts
│
├── dataset-generation/           # Synthetic data generation
│
├── models/
│   ├── forgetting_model.py       # ForgettingModel wraps forward call for p0
│   │
│   ├── encoder_prep/
│   │   └── encoder.py            # TextEncoder - T5 encoder wrapper
│   │
│   ├── slot_pooling_prep/
│   │   └── slot_pooling.py       # SlotPooling
│   │
│   ├── uv_heads_prep/
│   │   ├── u_head.py             # UHead - project to semantic anchor
│   │   └── v_head.py             # VHead - linear projection for detail latent
│   │
│   └── decoder_prep/
│       ├── decoder_x.py          # DecoderX - T5 decoder that reconstructs x from v
│       └── decoder_y.py          # DecoderY - T5 decoder that reconstructs y from u
│
├── denoiser/                     # Diffusion denoiser (Nθ)
│   ├── arch.md                   # Architecture specification
│   ├── config.py                 # Hyperparameters (L, d, T, N_blocks)
│   ├── denoiser.py               # Core: NoiseSchedule, Denoiser model, forward_diffusion, one_step_estimate equations
│   ├── trainer.py                # Training loop for denoiser
│   ├── inference.py              # Inference wrapper for recovery
│   ├── utils.py                  # Helper functions
│
└── data/                         # Training/validation data
```

---

## Component Details

### Data Loader

**Yet to be added**

### `models/encoder_prep/encoder.py` - TextEncoder

Wraps `T5EncoderModel` (**encoder-only**, no decoder). Takes tokenized text and returns hidden states.

| Property | Value |
|----------|-------|
| Base model | `t5-small` |
| Output | `[B, seq_len, 512]` |
| `hidden_dim_size` | 512 (from T5 config) |

---

### `models/slot_pooling_prep/slot_pooling.py` - SlotPooling

Compresses variable-length encoder output into a fixed number of slots using cross-attention.

| Input | Output |
|-------|--------|
| `H [B, seq, 512]` | `slots [B, 8, 512]` |

---

### `models/uv_heads_prep/u_head.py` - UHead

Produces the **semantic anchor `u`** - a compact vector summarizing the input.

`u` is used for:
- InfoNCE contrastive loss (paired with `upos`)
- DecoderY to reconstruct summary `y`
- Semantic conditioning in the denoiser(p1)

---

### `models/uv_heads_prep/v_head.py` - VHead

Produces the **detail latent `v0`** - preserves per-slot fine-grained information.

`v0` is used for:
- DecoderX to reconstruct original text `x`
- Input to the diffusion process in Stage C

---

### `models/decoder_prep/decoder_x.py` - DecoderX

Reconstructs original text `x` from detail latent `v0`.


---

### `models/decoder_prep/decoder_y.py` - DecoderY

Reconstructs summary `y` from semantic anchor `u`.

- Projects `u` from `[B, 128]` → `[B, 512]`
- Expands to `[B, 8, 512]` (repeats across slots)
- Feeds into T5 decoder as `encoder_outputs`
---

### `models/forgetting_model.py` - ForgettingModel

Wraps all components into a single `nn.Module` for clean training:

---

### `testing_refactored.py` - Refactored Training Script

Cleaner version using `ForgettingModel`. With added latent `u` and `v0` extraction.

---

### `denoiser/` - Diffusion Denoiser (Work in Progress)

The diffusion pipeline that operates on frozen `v0` and `u` latents from p0.

**The Flow**: `v0 → Forward Diffusion → vt → Denoiser Nθ(vt, t, u) → ε̂ → One-step Estimate → v0̂`

| Component | Description |
|-----------|-------------|
| `Denoiser` | 6-block Transformer with AdaLN, self-attention, cross-attention to `u`, FFN |
| `forward_diffusion()` | vt = √α̅·v0 + √(1-α̅)·ε |
| `one_step_estimate()` | v0̂ = (vt − √(1-α̅)·ε̂) / √α̅ |

Training: MSE loss between predicted noise `ε̂` and actual noise `ε`. Only the denoiser is trained; everything else is frozen.

---

## Two-Stage Training

### p0 Encoder + Heads + Decoders (current)

Train the full pipeline end-to-end:
- Encoder, SlotPooling, UHead, VHead, DecoderX, DecoderY
- All parameters trainable
- Three losses: InfoNCE + reconstruction x + reconstruction y

```bash
python testing_refactored.py
```

### Denoiser (after p0)

Freeze everything from p0. Train only the denoiser:
- Input: frozen `v0` (detail latent) and `u` (semantic anchor)
- Loss: MSE on noise prediction

```bash
cd denoiser/
python trainer.py
```

---

## Data Format

**yet to be added**

## Key Dimensions

| Tensor | Shape | Description |
|--------|-------|-------------|
| `H` | `[B, 64, 512]` | T5 encoder hidden states |
| `slots` | `[B, 8, 512]` | Slot-pooled representation |
| `u` | `[B, 128]` | Semantic anchor |
| `upos` | `[B, 128]` | Positive pair semantic anchor |
| `v0` | `[B, 8, 512]` | Detail latent |

---

## Requirements

`requirements.txt` for full list.
