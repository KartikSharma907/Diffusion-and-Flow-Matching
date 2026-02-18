# Diffusion & Flow Matching — Codebase Guide

A complete implementation of DDPM, DDIM, Flow Matching, and Flow Map Matching for image generation on CelebA-64. This README documents every component of the codebase in detail, with extra depth on Flow Map Matching, which is the method being extended next.

**Original course repo**: CMU 10799 Spring 2026 — Diffusion & Flow Matching
**Authors**: Yutong (Kelly) He, with assistance from Claude Code and OpenAI Codex

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [Data Pipeline](#data-pipeline)
4. [Model Architecture](#model-architecture)
   - [Building Blocks](#building-blocks-srcmodelsblockspy)
   - [U-Net](#u-net-srcmodelsunetpy)
   - [Dual-Time U-Net](#dual-time-u-net-srcmodelsdual_time_unetpy)
5. [Generative Methods](#generative-methods)
   - [Base Class](#base-class-srcmethodsbasepy)
   - [DDPM](#ddpm-srcmethodsddpmpy)
   - [DDIM Sampling](#ddim-sampling)
   - [Flow Matching](#flow-matching-srcmethodsflow_matchingpy)
   - [Flow Map Matching](#flow-map-matching-srcmethodsflow_map_matchingpy)
6. [Training Infrastructure](#training-infrastructure)
   - [Training Loop](#training-loop-trainpy)
   - [Exponential Moving Average](#exponential-moving-average-srcutilsemapy)
   - [Sampling Script](#sampling-script-samplepy)
7. [Configuration System](#configuration-system)
8. [Method Comparison](#method-comparison)
9. [Building on Flow Map Matching](#building-on-flow-map-matching)
10. [References](#references)

---

## Project Structure

```
Diffusion_flow_matching/
├── train.py                    # Main training script (all methods)
├── sample.py                   # Inference / sampling script
├── download_dataset.py         # CelebA dataset downloader from HuggingFace
├── data_analysis.py            # Dataset inspection utilities
├── modal_app.py                # Cloud GPU (Modal) integration
├── pyproject.toml              # Package metadata and dependencies
│
├── configs/
│   ├── ddpm.yaml               # DDPM training config
│   ├── ddpm_modal.yaml         # DDPM config for Modal cloud
│   ├── ddpm_v2_modal.yaml      # DDPM-v2 (x0-prediction) config
│   ├── flow_matching.yaml      # Flow Matching config
│   └── flow_map_matching.yaml  # Flow Map Matching config
│
├── src/
│   ├── models/
│   │   ├── blocks.py           # U-Net building blocks (ResBlock, Attention, etc.)
│   │   ├── unet.py             # Full U-Net architecture
│   │   └── dual_time_unet.py   # Dual-time wrapper for Flow Map Matching
│   ├── methods/
│   │   ├── base.py             # Abstract base class for all methods
│   │   ├── ddpm.py             # DDPM + DDIM implementation
│   │   ├── ddpm_v2.py          # DDPM variant that predicts x0 directly
│   │   ├── flow_matching.py    # Flow Matching (CondOT path)
│   │   └── flow_map_matching.py # Flow Map Matching (Lagrangian PDE)
│   ├── data/
│   │   └── celeba.py           # CelebA dataset loader and helpers
│   └── utils/
│       ├── ema.py              # Exponential Moving Average
│       └── logging_utils.py    # Logger and section headers
│
├── data/                       # Downloaded CelebA images
├── logs/                       # Training logs
├── checkpoints/                # Saved model checkpoints
├── notebooks/                  # Jupyter notebooks for exploration
├── scripts/                    # SLURM / evaluation shell scripts
└── docs/                       # Setup guides (SETUP.md, QUICKSTART-MODAL.md)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Download the CelebA-64 subset
python download_dataset.py --output_dir ./data/celeba-subset

# 3. Train DDPM
python train.py --config configs/ddpm.yaml

# 4. Train Flow Matching
python train.py --config configs/flow_matching.yaml

# 5. Train Flow Map Matching
python train.py --config configs/flow_map_matching.yaml

# 6. Sample from a DDPM checkpoint (1000 steps)
python sample.py --checkpoint checkpoints/ddpm/checkpoint_80000.pt \
                 --method DDPM --num_samples 64 --output samples_ddpm.png

# 7. Sample with DDIM acceleration (100 steps)
python sample.py --checkpoint checkpoints/ddpm/checkpoint_80000.pt \
                 --method DDIM --num_steps 100 --output samples_ddim.png

# 8. Sample from Flow Map Matching (20 steps)
python sample.py --checkpoint checkpoints/flow_map_matching/checkpoint_150000.pt \
                 --method FlowMapMatching --num_steps 20 --use_ema \
                 --output samples_fmm.png
```

---

## Data Pipeline

**File:** [src/data/celeba.py](src/data/celeba.py)

### Dataset: CelebA-64

The project trains on a 64×64 CelebA subset hosted on HuggingFace Hub (`electronickale/cmu-10799-celeba64-subset`). Images are loaded from either a local directory (default) or directly from the Hub.

### Normalization Convention

All images are normalized to the `[-1, 1]` range:
```
normalize:    pixel = (pixel / 255.0 - 0.5) / 0.5  →  [-1, 1]
unnormalize:  pixel = (pixel + 1.0) / 2.0           →  [0, 1]
```

This convention is required by diffusion models — Gaussian noise `N(0, I)` has values roughly in `[-3, 3]` with unit variance, which matches the scale of normalized data.

### `CelebADataset`

```python
class CelebADataset(Dataset):
    # Supports two loading modes:
    # 1. from_hub=True  → load directly from HuggingFace Hub
    # 2. from_hub=False → load from local directory of .jpg/.png files

    # Preprocessing pipeline:
    transforms = [
        RandomHorizontalFlip(),         # data augmentation (if augment=True)
        ToTensor(),                     # [0,255] uint8 → [0,1] float32
        Normalize([0.5]*3, [0.5]*3)     # [0,1] → [-1,1]
    ]
```

### `create_dataloader_from_config(config)`

Factory function that reads the `data:` section of a YAML config and creates the `DataLoader`. Key settings: `batch_size`, `num_workers`, `pin_memory`, `augment`.

### Helper Utilities

```python
unnormalize(x)         # [-1,1] tensor → [0,1] tensor (for saving images)
normalize(x)           # [0,1] tensor → [-1,1] tensor
make_grid(images, nrow) # Create an image grid from a batch
save_image(tensor, path) # Save a grid of images to disk as PNG
```

---

## Model Architecture

### Building Blocks ([src/models/blocks.py](src/models/blocks.py))

#### `SinusoidalPositionalEmbedding`

Encodes a scalar timestep `t` into a fixed-dimension vector using sinusoidal frequencies (identical to the positional encoding from the original Transformer paper):

```
freqs_k = exp(-log(10000) * k / (dim/2))   for k in 0 .. dim/2
emb     = [sin(t * freqs), cos(t * freqs)]  shape: (batch, dim)
```

Works for both integer timesteps (DDPM) and continuous values (Flow Matching / Flow Map Matching). The wide frequency range `[1, 10000]` ensures the embedding is unique for any timestep value.

#### `TimestepEmbedding`

A two-layer MLP that maps the raw sinusoidal encoding to a richer representation:

```
t → SinusoidalPositionalEmbedding(dim) → Linear(dim, 4*dim) → SiLU → Linear(4*dim, dim)
```

Output shape: `(batch, time_embed_dim)` where `time_embed_dim = base_channels * 4 = 512` for the default config. This embedding is injected into every `ResBlock` throughout the network.

#### `GroupNorm32`

Group normalization that always casts to `float32` before computing statistics, then casts back to the input dtype. This prevents numerical instability when using mixed precision (AMP), where activations may be in `float16`:

```python
def forward(self, x):
    return super().forward(x.float()).type(x.dtype)
```

#### `ResBlock`

The workhorse of the U-Net. Each block applies two 3×3 convolutions with a residual skip, injecting the time embedding between them:

```
x ──► GroupNorm32 ──► SiLU ──► Conv2d(in→out, 3×3) ──► h
                                                          │
                        time_emb ──► Linear(→out*2) ──► (scale, shift)
                                                          │
                       h ──► GroupNorm32 ──► FiLM(scale, shift) ──► SiLU ──► Dropout ──► Conv2d(out→out, 3×3)
                                                                                           │
x ──────────────────────────────────────────── [1×1 Conv if channels change, else Identity] ──► +
                                                                                                │
                                                                                            output
```

**FiLM Conditioning** (`use_scale_shift_norm=True`): The time embedding is projected to `2 * out_channels` and split into `scale` and `shift`. The second normalization becomes:
```
h = GroupNorm(h) * (1 + scale) + shift
```
This modulates every channel's gain and bias as a function of time, which is more expressive than simple addition.

**Simple Addition** (`use_scale_shift_norm=False`):
```
h = GroupNorm(h + time_emb)
```

#### `AttentionBlock`

Multi-head self-attention applied over all spatial locations. Applied only at resolutions in `attention_resolutions` (e.g., `[16, 8]`):

```
x → GroupNorm32 → QKV projection (1×1 Conv → 3*C channels)
  → reshape to (B, heads, H*W, head_dim)
  → attn = softmax(Q @ K^T / sqrt(d_head))
  → out = attn @ V
  → reshape back to (B, C, H, W)
  → 1×1 Conv output projection
  → residual add with x
```

Uses `einops.rearrange` for clean tensor reshaping. Scale factor `1/sqrt(head_dim)` prevents softmax saturation on large feature dimensions.

#### `Downsample` / `Upsample`

- **Downsample**: Strided 3×3 convolution (stride=2), halves H×W. Learned weights (vs. pooling) allow the network to optimize its own downsampling.
- **Upsample**: Nearest-neighbor interpolation (×2) followed by a 3×3 conv. More stable than transposed convolutions — avoids checkerboard artifacts.

---

### U-Net ([src/models/unet.py](src/models/unet.py))

The full encoder-decoder architecture with skip connections, standard for diffusion models.

#### Architecture (default config: `channel_mult=[1,2,2,4]`, `base_channels=128`, `num_res_blocks=2`)

```
Input: (B, 3, 64, 64)
│
└─ input_conv: Conv2d(3→128, 3×3)

Encoder (downsampling path):
  Level 0 │ 128ch │ 64×64 │
    ResBlock(128→128) │ skip → │ ResBlock(128→128) │ skip →
    Downsample(128)  ──► 32×32

  Level 1 │ 256ch │ 32×32 │
    ResBlock(128→256) │ skip → │ ResBlock(256→256) │ skip →
    Downsample(256)  ──► 16×16

  Level 2 │ 256ch │ 16×16 │  [attention at 16×16]
    ResBlock(256→256) + Attn │ skip → │ ResBlock(256→256) + Attn │ skip →
    Downsample(256)  ──► 8×8

  Level 3 │ 512ch │ 8×8   │  [attention at 8×8]
    ResBlock(256→512) + Attn │ skip → │ ResBlock(512→512) + Attn │ skip →
    Identity (no downsample at last level)

Middle: 512ch │ 8×8
    ResBlock(512→512)
    AttentionBlock(512)
    ResBlock(512→512)

Decoder (upsampling path):
  Level 3 │ 512ch │ 8×8   │  [attention at 8×8]
    ResBlock(512+512→512) + Attn │ ResBlock(512+512→512) + Attn
    Upsample(512)  ──► 16×16

  Level 2 │ 256ch │ 16×16 │  [attention at 16×16]
    ResBlock(512+256→256) + Attn │ ResBlock(256+256→256) + Attn
    Upsample(256)  ──► 32×32

  Level 1 │ 256ch │ 32×32 │
    ResBlock(256+256→256) │ ResBlock(256+256→256)
    Upsample(256)  ──► 64×64

  Level 0 │ 128ch │ 64×64 │
    ResBlock(256+128→128) │ ResBlock(128+128→128)
    Identity (no upsample at first level)

Output head:
    GroupNorm32(128) → SiLU → Conv2d(128→3, 3×3)

Output: (B, 3, 64, 64)
```

**Skip connections**: Each `ResBlock` in the encoder saves its output to a stack. Each `ResBlock` in the decoder pops from this stack and concatenates along the channel dimension before its conv. This is why decoder ResBlocks have `in_channels = ch + skip_ch`.

**Attention gating**: The `_maybe_attend` method checks if the current spatial resolution `H` is in `attention_resolutions`. Attention at 64×64 would be O(4096²) — prohibitively expensive. At 16×16 and 8×8 it is feasible.

#### `forward(x, t, time_emb=None)`

The optional `time_emb` argument is the key extension point for Flow Map Matching. When `None`, the internal `self.time_embedding(t)` is used. When provided (by `DualTimeUNet`), the precomputed combined `s+t` embedding bypasses internal embedding computation.

#### `create_model_from_config(config)`

Factory function reading the `model:` and `data:` sections from a YAML config, returning an instantiated `UNet`.

---

### Dual-Time U-Net ([src/models/dual_time_unet.py](src/models/dual_time_unet.py))

Flow Map Matching requires conditioning on **two** time values: source time `s` and target time `t`. `DualTimeUNet` wraps a standard `UNet` to support this.

#### Design

```
s ──► time_s_embedding (TimestepEmbedding) ──► s_emb ─┐
                                                        + ──► combined_emb ──► UNet.forward(x, t, time_emb=combined_emb)
t ──► unet.time_embedding (TimestepEmbedding) ──► t_emb ─┘

x ──────────────────────────────────────────────────────────────────────────► v_θ(s, t, x)
```

#### Key Implementation Notes

- **Scaling**: Continuous times `s, t ∈ [0, 1]` are multiplied by `num_timesteps` before being passed to the sinusoidal embedding. The sinusoidal embedding works better with magnitudes in the hundreds than near-zero decimals.
- **Combination**: The two time embeddings are **added** (not concatenated). Addition keeps the architecture unchanged — no new projection layers. Concatenation + projection is a valid alternative with more parameters.
- **No custom overrides needed**: PyTorch automatically handles `parameters()`, `state_dict()`, and `load_state_dict()` for submodules. Overriding these is a common mistake that breaks EMA, DDP, and checkpointing.

---

## Generative Methods

All methods inherit from `BaseMethod` and implement `compute_loss()` and `sample()`.

### Base Class ([src/methods/base.py](src/methods/base.py))

```python
class BaseMethod(nn.Module, ABC):
    # Abstract interface:
    def compute_loss(x, **kwargs) -> (loss_tensor, metrics_dict)  # must implement
    def sample(batch_size, image_shape, **kwargs) -> samples       # must implement

    # Provided:
    def train_mode() / eval_mode()
    def parameters()        # delegates to self.model.parameters()
    def state_dict()        # {'model': self.model.state_dict()}
    def load_state_dict()   # loads self.model
```

By inheriting from `nn.Module`, each method can register buffers via `self.register_buffer()`, be wrapped in `DistributedDataParallel`, and be moved with `.to(device)`.

---

### DDPM ([src/methods/ddpm.py](src/methods/ddpm.py))

**Paper:** Ho et al., 2020 — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

#### Conceptual Overview

DDPM defines a **discrete-time Markov chain** that gradually destroys data by adding Gaussian noise (the *forward process*), then trains a neural network to reverse this process (the *reverse process*). Sampling = running the reverse chain from pure noise to data.

#### Forward Process (Noise Schedule)

The forward process `q(x_{1:T} | x_0)` is a Markov chain:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) * x_{t-1}, β_t * I)
```

The **linear beta schedule** used here:
```
β_t = β_start + t * (β_end - β_start) / (T - 1)
β_start = 0.0001,  β_end = 0.02,  T = 1000
```

The key insight is the **reparameterized form** — sampling `x_t` from `x_0` in one step:
```
q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1-ᾱ_t) * I)
```
where `ᾱ_t = ∏_{i=1}^t αᵢ = ∏_{i=1}^t (1-β_i)` (cumulative product of signal retentions).

**Precomputed schedule buffers** (all stored as registered buffers, shape `[T]`):
```
betas                           β_t
alphas                          α_t = 1 - β_t
alphas_cumprod                  ᾱ_t = ∏ α_i
alphas_cumprod_prev             ᾱ_{t-1}  (prepended with ᾱ_0 = 1)
sqrt_alphas_cumprod             √ᾱ_t
sqrt_one_minus_alphas_cumprod   √(1-ᾱ_t)
sqrt_recip_alphas               √(1/α_t)
posterior_variance              σ²_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
posterior_log_variance_clipped  log(σ²_t)  (clipped for t=0 stability)
posterior_mean_coef1            β_t * √ᾱ_{t-1} / (1-ᾱ_t)
posterior_mean_coef2            (1-ᾱ_{t-1}) * √α_t / (1-ᾱ_t)
```

**`_extract(a, t, x_shape)`**: Indexes into a 1D buffer at indices `t` and reshapes to `(B, 1, 1, 1)` for broadcasting over image dimensions. Used throughout.

#### `forward_process(x_0, t)` → `(x_t, noise)`

```python
noise = randn_like(x_0)
x_t = sqrt_alphas_cumprod[t] * x_0 + sqrt_one_minus_alphas_cumprod[t] * noise
return x_t, noise
```

#### `compute_loss(x_0)` → `(loss, metrics)`

```python
t = randint(0, T, (B,))          # random timestep per sample
x_t, noise = forward_process(x_0, t)
predicted_noise = model(x_t, t)   # U-Net predicts the noise
loss = MSE(predicted_noise, noise)
```

This is the *simplified* Ho et al. objective. The full VLB objective is available but rarely used in practice.

#### `reverse_process(x_t, t)` → `x_{t-1}` (DDPM Sampling Step)

Given `x_t`, sample `x_{t-1}` from the posterior `q(x_{t-1} | x_t, x_0)`:

```
Step 1 — Predict noise:    ε_θ = model(x_t, t)
Step 2 — Predict x_0:     x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t    [clamped to [-1,1]]
Step 3 — Posterior mean:   μ = coef1 * x̂_0 + coef2 * x_t
Step 4 — Add noise:        x_{t-1} = μ + σ_t * z    (z = 0 when t = 0)
```

The `nonzero_mask = (t != 0).float()` ensures no noise is added at the final step.

#### `sample(...)` — Full DDPM Sampling Loop

```python
x_T ~ N(0, I)
for t in [T-1, T-2, ..., 1, 0]:
    x_{t-1} = reverse_process(x_t, t)
return x_0
```

Requires **T = 1000** network forward passes.

---

### DDIM Sampling

**Paper:** Song et al., 2020 — [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

#### Key Idea

DDIM discovers that the DDPM training objective is compatible with a **non-Markovian** reverse process, enabling:
1. **Deterministic sampling** (same noise → same image)
2. **Arbitrary subsampling** of the timestep schedule (100 steps instead of 1000)

The trick: instead of sampling from the full Markov posterior, DDIM directly predicts `x̂_0` and uses a fixed (non-stochastic with `η=0`) update.

#### `reverse_process_DDIM(x_t, t, t_prev, eta)` → `x_{t_prev}`

```
Step 1 — Predict noise:     ε_θ = model(x_t, t)
Step 2 — Predict x_0:       x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t    [clamped]
Step 3 — DDIM variance:     σ = η * √[(1-ᾱ_{t-1})/(1-ᾱ_t) * (1 - ᾱ_t/ᾱ_{t-1})]
Step 4 — Direction:         coeff = √(1 - ᾱ_{t-1} - σ²)
Step 5 — Update:            x_{t-1} = √ᾱ_{t-1} * x̂_0 + coeff * ε_θ + σ * z
```

When `t_prev = -1` (final step), the code forces `ᾱ_{t-1} = 1` so the step returns exactly `x̂_0`.

- **`η = 0`**: Fully deterministic. Same initial `x_T` always gives the same image.
- **`η = 1`**: Recovers DDPM stochasticity exactly.
- **`η ∈ (0,1)`**: Interpolates between deterministic and stochastic.

#### DDIM Sampling Schedule

```python
timesteps = linspace(0, T-1, num_steps).long()   # e.g., 100 evenly-spaced steps
timesteps = timesteps.flip(0)                      # reverse: [999, 989, ..., 9]
timesteps.append(-1)                               # sentinel for final x̂_0 step
```

With 100 steps: **10× speedup** over DDPM with comparable quality.

---

### Flow Matching ([src/methods/flow_matching.py](src/methods/flow_matching.py))

**Paper:** Lipman et al., 2022 — [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

#### Conceptual Overview

Flow Matching replaces the discrete Markov chain with a **continuous-time ODE**:
```
dX/dt = v_θ(X_t, t),    X(0) ~ N(0, I),  t ∈ [0, 1]
```

The model `v_θ` predicts the **velocity field** that transports noise (`t=0`) to data (`t=1`). Sampling = integrating this ODE forward with a numerical solver.

#### Conditional Optimal Transport (CondOT) Path

The simplest and most commonly used path is **linear interpolation** (also the optimal transport path between Gaussians and Diracs):

```
X_t = (1 - t) * X_0 + t * X_1
dX_t/dt = X_1 - X_0             (target velocity — constant for linear path)
```

where `X_0 ~ N(0, I)` is noise and `X_1 ~ p_data` is data. The velocity is **constant** along each sample path, making training stable and predictions easy.

#### `forward_process(x_0, x_1, t)` → `(x_t, velocity)`

```python
sigma_t = 1.0 - t      # noise weight (decreases over time)
alpha_t = t            # data weight (increases over time)
x_t = sigma_t * x_0 + alpha_t * x_1
velocity = x_1 - x_0  # constant target velocity
```

#### `compute_loss(x_1)` → `(loss, metrics)`

```python
noise = randn_like(x_1)                      # X_0 ~ N(0, I)
t = rand(B) * (1 - sigma_min) + sigma_min    # t ~ Uniform[sigma_min, 1]
x_t, target_v = forward_process(noise, x_1, t)
t_discrete = (t * (T-1)).long()               # scale t to [0, T-1] for model
predicted_v = model(x_t, t_discrete)
loss = MSE(predicted_v, target_v)             # = MSE(predicted_v, x_1 - x_0)
```

`sigma_min = 1e-4` avoids numerical issues at `t=0` where the signal is all noise.

#### `euler_step(x_t, t, dt)` → `x_{t+dt}`

```python
t_discrete = int(t * (T-1))
velocity = model(x_t, t_discrete)
x_next = x_t + dt * velocity      # Euler update
```

#### `sample(...)` — ODE Integration

```python
x = randn(B, C, H, W)          # start from noise at t=0
dt = 1.0 / num_steps

for step in range(num_steps):
    t = step / num_steps
    if method == "euler":
        x = euler_step(x, t, dt)
    elif method == "midpoint":
        x_mid = euler_step(x, t, dt/2)              # half step
        v_mid = model(x_mid, t_mid_discrete)         # velocity at midpoint
        x = x + dt * v_mid                           # full step with midpoint velocity
return x
```

The **midpoint method** (Runge-Kutta order 2) is more accurate than Euler for the same number of steps, at the cost of 2 model evaluations per step.

**Advantages over DDPM:**
- Straight-line paths in data space → faster convergence, fewer sampling steps needed
- ODE-based: deterministic, and can be reversed exactly
- Continuous time: no discrete schedule artifacts

---

### Flow Map Matching ([src/methods/flow_map_matching.py](src/methods/flow_map_matching.py))

**Paper:** Heitz et al., 2024 — [Flow Map Matching](https://arxiv.org/abs/2406.07507)

This is the central method being extended. It is considerably more sophisticated than DDPM or standard Flow Matching. Read carefully.

---

#### Core Concept: Flow Maps vs. Velocity Fields

Standard Flow Matching learns the **instantaneous velocity** `v_θ(x, t)` — what direction and speed to move at time `t`. This requires many small integration steps to transport noise to data accurately.

Flow Map Matching instead learns **flow maps** — functions that transport a state directly from one time to another:
```
Φ_{s,t}: x_s → x_t
```

A single flow map call can jump from `t=0` to `t=1` in one step (the ideal case). In practice, a small number of composed steps (5–20) gives excellent quality.

#### Flow Map Parameterization

The model parameterizes the flow map as:
```
Φ_{s,t}(x) = x + (t - s) * v_θ(s, t, x)
```

where `v_θ` is the neural network (a `DualTimeUNet`). This parameterization **automatically enforces the boundary condition** `Φ_{s,s}(x) = x` — when `t = s`, the correction term `(t-s) * v_θ = 0`, so the output equals the input exactly.

#### The Lagrangian PDE Constraint

A true flow map must satisfy the **Lagrangian PDE**:
```
∂_t Φ_{s,t}(x) = b_t(Φ_{s,t}(x))      (velocity at the transported point)
Φ_{s,s}(x) = x                          (boundary condition)
```

where `b_t` is the underlying velocity field of the ODE `dX/dt = b_t(X_t)`. The training objective enforces this PDE approximately via composition with the reverse map.

#### Deterministic Reference Path (Interpolant)

The ground-truth trajectory comes from the same linear interpolant as Flow Matching:
```
I_t = (1 - t) * x_0 + t * x_1       (position at time t)
İ_t = dI_t/dt = x_1 - x_0           (velocity — constant for linear path)
```

where `x_0 ~ N(0, I)` is noise and `x_1 ~ p_data` is data.

```python
def interpolant_I_t(t, x0, x1):
    return (1 - t) * x0 + t * x1

def interpolant_I_t_dot(t, x0, x1):
    return x1 - x0               # constant — independent of t
```

#### `flow_map(s, t, x)` → `Φ_{s,t}(x)`

```python
def flow_map(s, t, x):
    v_theta = self.model(x, s, t)           # DualTimeUNet: v_θ(s, t, x)
    delta = (t - s).view(-1, 1, 1, 1)
    return x + delta * v_theta              # Φ_{s,t}(x)
```

This is the fundamental building block. All sampling and loss computation go through this method.

---

#### Training: The Lagrangian Loss (Proposition 3.11)

The loss comes from Proposition 3.11 of the paper. The key idea: to enforce that `Φ_{s,t}` is consistent with the reference trajectory `I_t`, apply the **reverse map** `Φ_{t,s}` to bring `I_t` to the Lagrangian frame at time `s`, then apply `Φ_{s,t}` forward and check against `I_t`.

**Full loss:**
```
L = E_{s,t,x0,x1} [
    |∂_t Φ_{s,t}(Φ_{t,s}(I_t)) - İ_t|²    ← derivative matching
  + λ * |Φ_{s,t}(Φ_{t,s}(I_t)) - I_t|²   ← reconstruction
]
```

**Step-by-step implementation in `lagrangian_loss(s, t, I_t, I_t_dot)`:**

**Step 1: Reverse-direction map**
```python
X_ts_It = self.flow_map(t, s, I_t)          # Φ_{t,s}(I_t): move I_t backward from t to s
X_ts_It_fixed = X_ts_It.detach()            # detach — fixed Lagrangian base point
```
The detach is critical: we treat `Φ_{t,s}(I_t)` as a fixed point (the Lagrangian frame) and only differentiate through the forward composition `Φ_{s,t}`.

**Step 2: Compute `v_θ` and `∂_t v_θ` via JVP**

The time derivative of the flow map is computed analytically:
```
∂_t Φ_{s,t}(x) = v_θ(s,t,x) + (t-s) * ∂_t v_θ(s,t,x)
```

This needs `∂_t v_θ` — the Jacobian of the network output with respect to `t`. This is computed with `torch.func.jvp` (Jacobian-vector product), which computes the directional derivative in a **single forward pass** using forward-mode automatic differentiation:

```python
def model_single(t_single, s_single, x_single):
    # Wraps one model call for a single sample as a function of t
    return model(x.unsqueeze(0), s.unsqueeze(0), t.unsqueeze(0)).squeeze(0)

def compute_v_and_dv_dt(s_i, t_i, x_i):
    def fn(t_val):
        return model_single(t_val, s_i, x_i)
    # jvp computes (fn(t), ∂fn/∂t * 1) in one pass
    v_theta, dv_dt = jvp(fn, (t_i,), (torch.ones_like(t_i),))
    return v_theta, dv_dt

# Vectorize over the batch dimension using vmap
batched_jvp = vmap(compute_v_and_dv_dt, in_dims=(0, 0, 0), randomness="different")
v_theta, dv_dt = batched_jvp(s_fp32, t_fp32, X_ts_It_fp32)
```

**Why JVP and not backward AD?**
Computing `∂_t v_θ` via `loss.backward()` would require a second backward pass (expensive, O(n²) memory). `jvp` (forward-mode AD) computes the directional derivative `(d/dt)v_θ(s,t,x) * 1` in a **single forward pass** with no extra memory for backward. `vmap` then vectorizes this over the batch without Python loops.

**AMP / float32 requirement**: `torch.func.jvp` and `vmap` require float32 for numerical stability. The code explicitly wraps this section with `torch.amp.autocast(enabled=False)` and casts inputs to `.float()`.

**Step 3: Analytical time derivative of the flow map**
```python
delta = (t - s).view(-1, 1, 1, 1)
partial_t_X_st = v_theta + delta * dv_dt
```

**Step 4: Derivative matching loss**
```python
derivative_loss = MSE(partial_t_X_st, I_t_dot.float())
```

**Step 5: Reconstruction loss**
Compute `Φ_{s,t}(Φ_{t,s}(I_t))` — the forward map applied to the (non-detached) reverse map output:
```python
v_theta_rec = model(X_ts_It.float(), s_fp32, t_fp32)  # NOT detached
X_st_X_ts_It = X_ts_It.float() + delta * v_theta_rec
reconstruction_loss = MSE(X_st_X_ts_It, I_t.float())
```

Note: `X_ts_It` (not `X_ts_It_fixed`) is used here so that gradients flow through `Φ_{t,s}` as well.

**Total loss:**
```python
total_loss = derivative_loss + reconstruction_weight * reconstruction_loss
# reconstruction_weight = λ = 0.5 (from config)
```

---

#### Time Pair Sampling with Diagonal Annealing

Flow Map Matching trains on **pairs** `(s, t)` rather than a single timestep. The training distribution is the **triangle** `{(s,t) : 0 ≤ s < t ≤ 1}`.

A **curriculum** (diagonal annealing) is used for stable training: start with pairs close to the diagonal (`|t-s|` small, easy) and gradually expand to the full triangle (larger jumps, harder):

**`get_time_gap(step)`** — current maximum allowed `|t-s|`:

```
progress = min(1, step / warmup_steps)

linear:  max_gap(step) = initial_gap + (1 - initial_gap) * progress
cosine:  max_gap(step) = initial_gap + (1 - initial_gap) * (1 - cos(π*progress)) / 2
none:    max_gap(step) = 1.0  (no curriculum)
```

With `initial_gap=0.1` and `warmup_steps=30000`:
- **Step 0**: max `|t-s| = 0.1` (pairs very close to diagonal)
- **Step 15000**: max `|t-s| ≈ 0.55`
- **Step 30000+**: max `|t-s| = 1.0` (full triangle)

**Why curriculum?** When `s ≈ t`, the flow map is nearly the identity (`Φ_{s,t}(x) ≈ x`), so the loss signal is clean and the gradients are well-conditioned. Starting close to the diagonal lets the network learn local structure, then generalize to large jumps.

**`sample_time_pairs(batch_size)`** — samples from the triangle with a triangular density:

```
σ = sigma_min
gap = current_max_gap

# Sample δ = t - s from p(δ) ∝ (gap - δ) on [σ, gap]
# (triangular: more mass near δ=0, less near δ=gap)
# Inverse CDF: δ = gap - (gap - σ) * √(1-u),  u ~ Uniform[0,1]
u ~ Uniform[0, 1]
δ = gap - (gap - σ) * √(1 - u)

# Sample s uniformly given δ, within the feasible interval
s ~ Uniform[σ, max_gap - δ]
t = s + δ
```

The triangular density `p(δ) ∝ (gap - δ)` is the marginal distribution of `δ` when `(s, t)` is uniform over the triangle — it naturally downweights large gaps where less training area exists.

---

#### Sampling: Sequential Composition

At inference, learned flow maps are **composed sequentially** to transport noise to data:

```
x_0 ~ N(0, I)
t_0 = σ_min,  t_1, ..., t_N = 1 - σ_min    (N+1 evenly-spaced timepoints)

x_1 = Φ_{t_0, t_1}(x_0)          # step 1: noise → slightly less noisy
x_2 = Φ_{t_1, t_2}(x_1)          # step 2
...
x_N = Φ_{t_{N-1}, t_N}(x_{N-1})  # step N: data
```

```python
def sample(batch_size, image_shape, num_steps=20):
    x = randn(batch_size, *image_shape)
    ts = linspace(sigma_min, 1 - sigma_min, num_steps + 1)
    for i in range(num_steps):
        s = ts[i].expand(batch_size)
        t = ts[i+1].expand(batch_size)
        x = self.flow_map(s, t, x)     # Φ_{s,t}(x)
    return x
```

With `num_steps=20`, this is only **20 neural network forward passes** — much fewer than DDPM (1000) or Flow Matching (100). The learned flow maps can span large time intervals since they are trained on the full triangle.

---

#### `compute_loss` — Complete Pipeline

```python
def compute_loss(x_1):
    x_0 = randn_like(x_1)                         # noise
    s, t = sample_time_pairs(batch_size)           # diagonal-annealed pairs
    I_t = interpolant_I_t(t, x_0, x_1)            # reference path position
    I_t_dot = interpolant_I_t_dot(t, x_0, x_1)   # reference path velocity
    loss, loss_dict = lagrangian_loss(s, t, I_t, I_t_dot)
    return loss, metrics
```

**Metrics logged:** `loss`, `s_mean`, `t_mean`, `delta_t_mean`, `current_gap`, `derivative_loss`, `reconstruction_loss`.

---

## Training Infrastructure

### Training Loop ([train.py](train.py))

The training script handles all four methods from a single entry point via `--config`. Key features:

#### Setup

```python
config = load_config(args.config)
model = create_model_from_config(config)
# For FlowMapMatching: model = DualTimeUNet(base_unet, num_timesteps)
method = DDPM / FlowMatching / FlowMapMatching.from_config(model, config, device)
optimizer = AdamW(method.parameters(), lr=lr, betas=betas, weight_decay=wd)
ema = EMA(model, decay=ema_decay, warmup_steps=ema_start)
scaler = GradScaler()                    # mixed precision gradient scaler
```

#### Multi-GPU via DistributedDataParallel

```python
if num_gpus > 1:
    method = DDP(method, device_ids=[local_rank])
```

DDP wraps the entire `BaseMethod`. `FlowMapMatching._func_model()` unwraps the `.module` attribute when needed inside `torch.func` transforms.

#### Training Step with Mixed Precision

```python
with torch.amp.autocast(device_type='cuda', enabled=mixed_precision):
    loss, metrics = method.compute_loss(batch)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(method.parameters(), gradient_clip_norm)
scaler.step(optimizer)
scaler.update()
ema.update()
```

Gradient clipping (default `clip_norm=1.0`) prevents gradient explosions, which are common with the JVP computation in Flow Map Matching.

**AMP + JVP**: The `lagrangian_loss` function internally disables autocast for the JVP section. The outer `autocast` context is safe to keep enabled — it applies to all other operations including the reverse-direction map `Φ_{t,s}(I_t)`.

#### Single-Batch Overfitting Mode

```bash
python train.py --config configs/flow_map_matching.yaml --overfit
```

Fixes one batch and trains on it repeatedly. Useful for debugging: the loss should reach near-zero, and outputs should look like the training images. If the loss doesn't decrease, there is a bug.

#### Checkpointing

```python
checkpoint = {
    'iteration': iteration,
    'model': method.state_dict(),            # model weights
    'optimizer': optimizer.state_dict(),     # optimizer state
    'ema': ema.state_dict(),                 # EMA shadow weights
    'config': config,                        # full YAML config
    'method_config': method_specific_config, # method hyperparameters
}
torch.save(checkpoint, path)
```

#### Periodic Sampling During Training

Every `sample_every` iterations, the script generates a grid of samples (using EMA weights when available) and saves them to the log directory. This is the primary way to monitor training progress.

---

### Exponential Moving Average ([src/utils/ema.py](src/utils/ema.py))

EMA maintains a **shadow copy** of model parameters, updated after each optimizer step:
```
θ_ema ← decay * θ_ema + (1 - decay) * θ_model
```

With `decay = 0.9999`, the EMA lags behind the current model by roughly `1/(1-0.9999) = 10,000` steps. This smooths out stochastic gradient noise and produces significantly cleaner samples at inference.

#### Warmup

During early training, a smaller effective decay is used so the EMA adapts quickly:
```python
def get_decay(self):
    if self.step < warmup_steps:
        return min(self.decay, (1 + self.step) / (10 + self.step))
    return self.decay
```
At step 0: effective decay ≈ 0.09 (fast adaptation).
At step 100: decay approaches the configured value.

#### Usage Pattern

```python
# After each optimizer step:
ema.update()

# For sampling (swap in EMA weights):
ema.apply_shadow()
samples = method.sample(...)
ema.restore()              # swap back training weights
```

---

### Sampling Script ([sample.py](sample.py))

```bash
python sample.py \
    --checkpoint checkpoints/flow_map_matching/checkpoint_150000.pt \
    --method FlowMapMatching \
    --num_samples 64 \
    --num_steps 20 \
    --use_ema \
    --output samples_fmm.png
```

**Key features:**
- `strip_module_prefix()` — removes the `module.` prefix added by DDP wrapping in saved checkpoints
- `unwrap_state_dict()` — handles various checkpoint nesting patterns for compatibility
- `--use_ema` — loads EMA shadow weights (always preferred for final samples)
- Method-specific arguments forwarded to `method.sample()` via `**kwargs`

**Method-specific flags:**
- DDPM: no extra flags (defaults to 1000 steps)
- DDIM: `--method DDIM --num_steps 100 --eta 0.0`
- Flow Matching: `--method FlowMatching --num_steps 100`
- Flow Map Matching: `--method FlowMapMatching --num_steps 20`

---

## Configuration System

All experiments use YAML configs. Every config has the same top-level structure:

```yaml
data:
  dataset: "celeba"
  root: /path/to/celeba-subset   # local path
  from_hub: false                # set true to load from HuggingFace Hub
  repo_name: "electronickale/cmu-10799-celeba64-subset"
  image_size: 64
  channels: 3
  num_workers: 4
  pin_memory: true
  augment: true                  # random horizontal flip

model:
  base_channels: 128
  channel_mult: [1, 2, 2, 4]    # → [128, 256, 256, 512] channels per level
  num_res_blocks: 2
  attention_resolutions: [16, 8] # apply attention at 16×16 and 8×8
  num_heads: 4
  dropout: 0.1                   # 0.0 for Flow Map Matching
  use_scale_shift_norm: true     # FiLM conditioning in ResBlocks

training:
  batch_size: 128                # per-GPU batch size
  learning_rate: 0.0002
  weight_decay: 0.0001
  betas: [0.9, 0.999]            # AdamW momentum parameters
  ema_decay: 0.9999
  ema_start: 10000               # steps before EMA begins (0 = immediate)
  gradient_clip_norm: 1.0
  num_iterations: 80000          # 150000 for Flow Map Matching
  log_every: 100
  sample_every: 1000
  save_every: 5000
  num_samples: 16                # grid size for training samples

# Method-specific block (only one is used per config):
ddpm:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02

flow_matching:
  num_timesteps: 1000
  sigma_min: 0.0001

flow_map_matching:
  num_timesteps: 1000
  sigma_min: 0.0001
  annealing_schedule: "linear"  # "linear", "cosine", or "none"
  initial_gap: 0.1              # starting max |t-s|
  warmup_steps: 30000           # steps to reach full triangle
  reconstruction_weight: 0.5    # λ in Prop. 3.11

sampling:
  num_steps: 20                  # 1000 for DDPM, 100 for FM, 20 for FMM
  batch_size: 16

infrastructure:
  seed: 42
  device: "cuda"
  num_gpus: 2
  mixed_precision: true          # set false if NaN losses with torch.func
  compile_model: false

checkpoint:
  dir: "./checkpoints"
  resume: null                   # path to resume from, or null

logging:
  dir: "./logs"
  wandb:
    enabled: true
    project: "cmu-10799-diffusion"
    entity: null
```

---

## Method Comparison

| Property | DDPM | DDIM | Flow Matching | Flow Map Matching |
|----------|------|------|---------------|-------------------|
| **Time domain** | Discrete [0, T] | Discrete [0, T] | Continuous [0, 1] | Continuous [0, 1] |
| **Path type** | Markov chain | Non-Markovian | Linear ODE | Lagrangian flow map |
| **Training target** | Noise `ε` | Noise `ε` | Velocity `v = x₁-x₀` | Velocity `v_θ(s,t,x)` |
| **Network input** | `(x_t, t)` | `(x_t, t)` | `(x_t, t)` | `(x, s, t)` — dual time |
| **Network arch** | UNet | UNet | UNet | DualTimeUNet |
| **Loss** | `‖ε_θ - ε‖²` | `‖ε_θ - ε‖²` | `‖v_θ - (x₁-x₀)‖²` | Lagrangian PDE loss (JVP) |
| **Training cost** | Low | Low | Low | High (JVP + vmap) |
| **Sampling steps** | ~1000 | ~100 | ~100 | ~5–20 |
| **Sampling method** | Markov chain | ODE (deterministic) | Euler/Midpoint ODE | Sequential composition |
| **Deterministic?** | No | Yes (`η=0`) | Yes | Yes |
| **Stochastic?** | Yes | Yes (`η>0`) | No | No |
| **Key advantage** | Simple, stable | Fast DDPM sampling | Straight-line paths | Fewest sampling steps |
| **Key challenge** | 1000 steps required | Slightly lower quality | Moderate steps | Complex training (JVP) |

---

## Building on Flow Map Matching

The `FlowMapMatching` class is carefully designed to be extended. Here are the key extension points, invariants, and practical considerations.

### Invariants to Preserve

1. **Boundary condition**: `flow_map(s, s, x) = x` always holds because `(t-s) = 0` in the parameterization `x + (t-s)*v_θ`. Do not change the parameterization without preserving this.

2. **AMP safety**: Any code that uses `torch.func.jvp` or `vmap` must run in float32. Wrap with `torch.amp.autocast(device_type='cuda', enabled=False)` and cast inputs to `.float()`.

3. **DDP compatibility**: Use `self._func_model()` (not `self.model`) inside `vmap`/`jvp`. `_func_model()` unwraps the DDP `.module` attribute when needed, since `torch.func` transforms cannot be vmapped over DDP-wrapped modules.

4. **Step counter**: `self._step` is a registered buffer (so it survives checkpointing) and is incremented inside `sample_time_pairs` only when `self.training` is True. Do not increment it elsewhere.

### Architecture Extension Points

**`flow_map(s, t, x)`**
The core method. Changing the parameterization here affects everything. Example: replace linear with a more expressive parameterization:
```python
# Current:
return x + (t - s) * v_theta
# Alternative (exponential map):
return x * exp((t-s) * v_theta)   # different geometry
```

**`interpolant_I_t` / `interpolant_I_t_dot`**
Currently uses the linear CondOT path. Can be replaced with stochastic interpolants (add a diffusion term), bridge processes, or other paths. Note: `I_t_dot` must be the time derivative of `I_t` — update both consistently.

**`sample_time_pairs`**
Controls the training distribution over `(s, t)`. Can be replaced with importance sampling (oversample hard pairs where the current model has high loss), stratified sampling (guarantee coverage of the triangle), or non-uniform schedules.

**`lagrangian_loss`**
Implements Prop. 3.11 with `w_{s,t} = 1`. Can add adaptive weighting:
```python
weight = compute_weight(s, t, loss_values)    # e.g., higher weight for hard pairs
derivative_loss = (weight * (∂_t_X - I_t_dot)**2).mean()
```

**`get_time_gap` / annealing_schedule**
Currently supports "linear", "cosine", and "none". Easy to add "exponential" or step-function schedules by adding cases in `get_time_gap`.

### Adding Class or Text Conditioning

To condition on class labels or other signals:
1. Update `DualTimeUNet.forward(x, s, t, cond=None)` to accept and process conditioning
2. Pass conditioning to `UNet.forward` (e.g., via cross-attention or additional embedding)
3. Update `flow_map(s, t, x, cond=None)` to forward conditioning to the model
4. Update `compute_loss` to sample conditioning labels and pass them through

### Changing the Reconstruction Loss

The current reconstruction term is a direct identity composition `Φ_{s,t}(Φ_{t,s}(I_t))`. This can be replaced with multi-step compositions or other consistency constraints:

```python
# Example: multi-step consistency loss
u = (s + t) / 2
X_su = flow_map(s, u, x)       # s → u
X_ut = flow_map(u, t, X_su)    # u → t
X_st = flow_map(s, t, x)       # s → t directly
consistency_loss = MSE(X_ut, X_st)   # should be equal
```

### Stochastic Interpolants

Replace the deterministic linear path with a stochastic interpolant that adds diffusion noise:
```python
def interpolant_I_t(t, x0, x1, z=None):
    if z is None:
        z = randn_like(x0)
    gamma_t = sigma(t)            # noise level schedule
    return (1-t)*x0 + t*x1 + gamma_t * z
```
`interpolant_I_t_dot` would then include the Brownian motion term's time derivative.

### Config Extension Pattern

New hyperparameters should follow this pattern:
1. Add to `flow_map_matching:` section of the YAML
2. Read in `from_config` with a default: `fm_config.get("new_param", default_value)`
3. Store as `self.new_param = new_param` in `__init__`
4. Do **not** add to `state_dict` unless it's a tensor/buffer — scalar hyperparameters are saved in the checkpoint's `method_config` key automatically by the training script

### Performance Considerations

- `vmap` + `jvp` is the bottleneck: it scales linearly with batch size but uses forward-mode AD, which has ~2× overhead vs. a regular forward pass.
- Setting `mixed_precision: false` in the config removes the float32 casting overhead around the JVP section, at the cost of overall training throughput.
- Reducing `batch_size` (e.g., 16) reduces memory pressure from the dual forward pass in `lagrangian_loss`.
- The `derivative_loss` and `reconstruction_loss` can be monitored separately to diagnose training issues: if reconstruction is near zero but derivative is high, the model is learning to compose but not satisfy the PDE locally.

---

## References

- **DDPM**: Ho, J., Jain, A., Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS 2020.
- **DDIM**: Song, J., Meng, C., Ermon, S. (2020). [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502). ICLR 2021.
- **Flow Matching**: Lipman, Y., et al. (2022). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR 2023.
- **Flow Map Matching**: Heitz, E., et al. (2024). [Flow Map Matching](https://arxiv.org/abs/2406.07507).
- **U-Net / ADM**: Dhariwal, P., Nichol, A. (2021). [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233). NeurIPS 2021.
- **FiLM**: Perez, E., et al. (2018). [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871). AAAI 2018.
