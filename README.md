# ITOPipeline: SDXL with Information-Theoretic Guidance

This repo provides a thin, production‑oriented wrapper around
[`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)
that adds:

- **Information-Theoretic Optimization (ITO) guidance**  
  – adaptive per‑step guidance strength based on a **KL budget**
- **Stable FP32 decoding**  
  – VAE runs in float32 with tiling/slicing to reduce NaNs and OOMs
- **Standard SDXL path with fp32 decode**  
  – a `generate_fixed` helper for “vanilla” SDXL images
- **Negative prompts & SDXL dual‑prompt support**
- A small **`make_grid`** utility to tile images with labels

The goal is to make it easy to A/B test ITO guidance vs. vanilla SDXL in a
single, clean interface.

---

## 1. Installation

You’ll need:

- Python 3.9+
- PyTorch with CUDA (recommended) or CPU
- `diffusers`, `transformers`, `accelerate`, `safetensors`, `Pillow`, `numpy`

Example install:

```bash
pip install torch diffusers transformers accelerate safetensors pillow numpy
```

> ⚠️ Make sure your PyTorch build matches your CUDA version if you’re on GPU.

By default the pipeline loads:

- `stabilityai/stable-diffusion-xl-base-1.0`

You may need a Hugging Face token configured to download the model.

---

## 2. What is ITO guidance?

Instead of using a fixed CFG scale (e.g. 7.5) for all diffusion steps,
**ITO guidance**:

- Treats each diffusion step as spending part of a **KL budget**.
- Measures how different the conditional and unconditional noise predictions are.
- Chooses a **per‑step guidance strength** `lambda_k` so that:

  \[
  \sum_k rac{1}{2} \lambda_k^2 q_k pprox 	ext{budget}
  \]

  where `q_k` is a scalar measure of how strong the text signal is at step `k`.

This means:

- **Early steps** (high noise, strong text signal) tend to get **higher λ**.
- **Later steps** (low noise, weaker text signal) tend to get **lower λ**.
- You control how “hard” the model is pushed overall via a single `budget`
  (instead of guessing a fixed CFG scale).

On top of that, there’s an **`alpha` soft-rescale** that stabilizes guidance by
matching the variance of the guided noise to the “pure text” direction.

---

## 3. Quickstart

Assuming your main file is called `ito_pipeline.py` and contains `ITOPipeline`:

```python
from ito_pipeline import ITOPipeline

# Create the pipeline (loads SDXL weights)
ito = ITOPipeline()

prompt = "a cozy cabin in a snowy forest, warm lights in the windows"

# --- ITO-guided generation ---
image_ito, total_kl, lambdas = ito.generate_ito(
    prompt=prompt,
    negative_prompt="low quality, blurry, distorted, text",
    budget=40.0,
    lambda_max=7.5,
    num_steps=40,
    height=1024,
    width=1024,
    alpha=0.3,
    seed=42,
    verbose=False,
)
image_ito.save("cabin_ito.png")

# --- Baseline SDXL with fixed CFG ---
image_fixed = ito.generate_fixed(
    prompt=prompt,
    negative_prompt="low quality, blurry, distorted, text",
    guidance_scale=7.5,
    num_steps=40,
    height=1024,
    width=1024,
    seed=42,
)
image_fixed.save("cabin_fixed.png")
```

You can then compare `cabin_ito.png` vs `cabin_fixed.png`.

---

## 4. ITOPipeline API

### 4.1 Initialization

```python
ito = ITOPipeline(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",  # default
)
```

What happens under the hood:

- Loads `StableDiffusionXLPipeline.from_pretrained(model_id, ...)`
  - Uses `torch.float16` on GPU / MPS, `torch.float32` on CPU
  - Enables `safetensors`
- Replaces the scheduler with **DPM-Solver++ (Karras sigmas)**.
- Moves the **VAE to float32** and enables:
  - `enable_vae_tiling()`
  - `enable_vae_slicing()`
- Enables TF32 matmul on CUDA for a performance/precision sweet spot.
- Disables the diffusers progress bar (less noisy logs).

---

### 4.2 `generate_ito(...)`

```python
image, total_kl, lambdas = ito.generate_ito(
    prompt: str,
    prompt_2: Optional[str] = None,
    budget: float = 40.0,
    lambda_max: Optional[float] = 7.5,
    num_steps: int = 40,
    seed: int = 42,
    height: int = 1024,
    width: int = 1024,
    alpha: float = 0.3,
    verbose: bool = True,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
)
```

**Key parameters:**

- `prompt`: main SDXL text prompt.
- `prompt_2`: optional second SDXL text encoder prompt (e.g. style / extra info).
- `negative_prompt`, `negative_prompt_2`: standard SDXL negatives for each encoder.
- `budget`: total KL budget. Higher = stronger overall guidance.
- `lambda_max`: hard per‑step cap on guidance strength. `None` disables the cap.
- `num_steps`: diffusion steps (e.g. 30–50).
- `height`, `width`: resolution in pixels (multiples of 64 recommended).
- `alpha`:
  - `0.0` → minimal rescale (more vivid / aggressive guidance)
  - `1.0` → fully rescaled (more stable, closer to standard CFG behavior)
- `seed`: for reproducibility.
- `verbose`: print per‑10‑step stats, final KL and latent/decoded stats.

**Returns:**

- `image` — `PIL.Image.Image`, decoded from fp32 VAE.
- `total_kl` — accumulated KL usage, approx. `<= budget`.
- `lambdas` — list of per‑step guidance strengths used.

---

### 4.3 `generate_fixed(...)`

```python
image = ito.generate_fixed(
    prompt: str,
    prompt_2: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_steps: int = 40,
    seed: int = 42,
    height: int = 1024,
    width: int = 1024,
)
```

This is a convenience wrapper for **standard SDXL generation**, but:

- Asks diffusers to output **latents** (`output_type="latent"`).
- Decodes those latents using the **fp32 VAE** in this pipeline.
- Clamps VAE output to `[-1, 1]` and runs `image_processor` to get a PIL image.

Use this to:

- Get a “baseline” SDXL image for comparison with the ITO version.
- Enjoy more stable VAE decoding than the default fp16 path.

---

### 4.4 `make_grid(images, labels)`

```python
from ito_pipeline import make_grid

grid = make_grid(
    images=[image_ito, image_fixed],
    labels=["ITO (budget 40)", "Baseline CFG 7.5"],
)
grid.save("comparison_grid.png")
```

- Creates a horizontal grid of images with 40px of padding at the top.
- Draws the corresponding label above each image (if provided).

---

## 5. Example: Comparing ITO vs Fixed CFG

```python
from ito_pipeline import ITOPipeline, make_grid

ito = ITOPipeline()

prompt = "a futuristic cityscape at dusk, cinematic lighting, ultra-detailed"
neg = "low quality, blurry, distorted, text, watermark"

image_ito, total_kl, lambdas = ito.generate_ito(
    prompt=prompt,
    negative_prompt=neg,
    budget=40.0,
    lambda_max=7.5,
    num_steps=30,
    alpha=0.25,
    height=1024,
    width=1024,
    seed=1234,
    verbose=False,
)

image_cfg = ito.generate_fixed(
    prompt=prompt,
    negative_prompt=neg,
    guidance_scale=7.5,
    num_steps=30,
    height=1024,
    width=1024,
    seed=1234,
)

grid = make_grid(
    images=[image_ito, image_cfg],
    labels=[f"ITO (KL={total_kl:.1f})", "CFG=7.5"],
)
grid.save("ito_vs_cfg.png")
```

---

## 6. Tips & Notes

- **GPU strongly recommended.** SDXL at 1024×1024 on CPU is extremely slow.
- If you hit **NaNs / infs** in latents, the code already:
  - Detects non‑finite values.
  - Sanitizes them with `torch.nan_to_num`.
- If you get **OOMs**, try:
  - Lowering `height`/`width` (e.g. to 768×768).
  - Reducing `num_steps`.
- `budget`, `lambda_max`, and `alpha` are meant to be **tunable knobs**:
  - Start with `budget=30–50`, `lambda_max=7.5`, `alpha=0.3`.
  - Increase `budget` for more “locked‑in” prompt adherence (at risk of artifacts).
  - Increase `alpha` for more stability / less aggressive guidance.

---

## 7. License / Attribution

This repo **wraps** the `StableDiffusionXLPipeline` and SDXL model weights
provided by Stability AI via Hugging Face. Please make sure your usage complies
with:

- The license terms of the SDXL model you load (e.g. `stabilityai/stable-diffusion-xl-base-1.0`).
- Hugging Face’s and Stability AI’s usage policies.

The code in this repo can be used and modified under whatever license you attach
to it (fill this section in as appropriate for your project).
