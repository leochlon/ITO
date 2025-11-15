import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional


class ITOPipeline:
    """
    SDXL pipeline with Information-Theoretic Optimization (ITO) style guidance.

    Methods
    -------
    generate_ito(...)
        ITO-guided generation with per-step KL budget.
    generate_fixed(...)
        Standard SDXL generation with fixed guidance scale.
    """

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0") -> None:
        # Device & dtype setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        variant = "fp16" if model_dtype == torch.float16 else None

        # Base SDXL pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            variant=variant,
            use_safetensors=True,
        ).to(self.device)

        # Scheduler: DPM-Solver++ with Karras sigmas
        from diffusers import DPMSolverMultistepScheduler

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

        # Decode in float32 for stability
        self.pipe.vae.to(dtype=torch.float32)
        self.pipe.enable_vae_tiling()
        self.pipe.enable_vae_slicing()

        # Allow TF32 on CUDA (safe no-op on CPU builds, but we guard anyway)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        # No progress bar by default (better for prod logs)
        self.pipe.set_progress_bar_config(disable=True)

    def compute_q_k(self, noise_pred_cond: torch.Tensor, noise_pred_uncond: torch.Tensor) -> float:
        """
        Convenience helper: simple squared-diff metric between conditional and unconditional noise.
        Not used in the main loop, but kept for external debugging/analysis.
        """
        diff = noise_pred_cond - noise_pred_uncond
        q = torch.sum(diff ** 2).item()
        return q

    @torch.no_grad()
    def generate_ito(
        self,
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
    ) -> Tuple[Image.Image, float, List[float]]:
        """
        ITO-guided SDXL sampling.

        Parameters
        ----------
        prompt : str
            Main text prompt.
        prompt_2 : str, optional
            Optional second SDXL text encoder prompt.
        budget : float
            Total KL budget to distribute across steps.
        lambda_max : float or None
            Per-step maximum for lambda (guidance strength). If None, no cap.
        num_steps : int
            Number of diffusion steps.
        seed : int
            RNG seed for reproducibility.
        height, width : int
            Output image size in pixels.
        alpha : float
            Soft guidance rescale factor in [0, 1]. Lower = more vivid, higher = more stable.
        verbose : bool
            If True, prints basic diagnostics.
        negative_prompt : str, optional
            Negative prompt for the first text encoder.
        negative_prompt_2 : str, optional
            Negative prompt for the second SDXL text encoder.

        Returns
        -------
        image : PIL.Image.Image
            Generated image.
        total_kl : float
            Approximate accumulated KL across steps.
        lambdas : List[float]
            Per-step guidance strengths.
        """
        generator = torch.Generator(self.device).manual_seed(seed)

        # --- Scheduler setup & latents (FP32 master) ---
        self.pipe.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        h_lat = height // self.pipe.vae_scale_factor
        w_lat = width // self.pipe.vae_scale_factor

        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, h_lat, w_lat),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

        # Scale initial noise by scheduler sigma
        latents = latents * self.pipe.scheduler.init_noise_sigma

        # --- Encode prompts (model dtype on GPU, fp32 on CPU) ---
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
        )

        text_encoder_projection_dim = getattr(self.pipe.text_encoder_2.config, "projection_dim", None)
        add_time_ids = self.pipe._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(self.device)

        # Duplicate conditionals for classifier-free guidance
        prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds_combined = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        add_time_ids_combined = torch.cat([add_time_ids, add_time_ids], dim=0)

        # For determinism with schedulers that use extra noise
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator=generator, eta=0.0)

        # Pre-fetch sigmas if present
        sigmas = getattr(self.pipe.scheduler, "sigmas", None)
        if sigmas is not None and hasattr(sigmas, "to"):
            sigmas = sigmas.to(self.device)

        # --- Distribute KL budget with cosine front-loading ---
        w = (1 - torch.cos(torch.linspace(0, torch.pi, len(timesteps), device=self.device))) / 2
        w = w / w.sum()
        kappas = (budget * w).tolist()

        if verbose:
            print(f"Budget: {budget:.1f}, Lambda_max: {lambda_max:.2f}")

        total_kl: float = 0.0
        lambdas: List[float] = []

        # --- Main sampling loop ---
        for i, t in enumerate(timesteps):
            # Keep master latents in FP32 for scheduler math
            latent_model_input = self.pipe.scheduler.scale_model_input(latents, t)

            # UNet expects same dtype as weights (fp16/fp32)
            latent_unet = torch.cat([latent_model_input, latent_model_input], dim=0).to(self.pipe.unet.dtype)

            added_cond_kwargs = {
                "text_embeds": add_text_embeds_combined,
                "time_ids": add_time_ids_combined,
            }

            # UNet forward
            noise_pred_combined = self.pipe.unet(
                latent_unet,
                t,
                encoder_hidden_states=prompt_embeds_combined,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2, dim=0)

            # Compute q_k in float32 and scale by scheduler variance if available
            diff = (noise_pred_cond - noise_pred_uncond).float()
            if sigmas is not None:
                scale = float(sigmas[i].item() ** 2)
            else:
                scale = 1.0
            q_k = (diff.pow(2).mean() * scale).item()

            # Numerically safe lambda_k
            denom = q_k + 1e-12
            kappa_k = kappas[i]
            lambda_k = float(np.sqrt(max(0.0, (2.0 * kappa_k) / denom)))
            if lambda_max is not None:
                lambda_k = min(lambda_k, float(lambda_max))

            lambdas.append(lambda_k)

            if verbose and (i % 10 == 0 or i == len(timesteps) - 1):
                print(f"Step {i:2d}: lambda={lambda_k:.3f}, q_k={q_k:.4e}")

            # Guided noise (back to FP32 for scheduler)
            noise_pred = (
                noise_pred_uncond + lambda_k * (noise_pred_cond - noise_pred_uncond)
            ).to(torch.float32)

            # --- Soft guidance rescale (alpha) ---
            diff_text = (noise_pred_cond - noise_pred_uncond).float()
            diff_guided = (noise_pred - noise_pred_uncond).float()

            std_text = diff_text.std(unbiased=False)
            std_guided = diff_guided.std(unbiased=False)

            eps = 1e-6
            if std_guided > eps and std_text > 0:
                # alpha in [0, 1]; lower = more vivid, higher = more stable
                scale_rescale = (1.0 - alpha) + alpha * (std_text / std_guided)
                noise_pred = diff_guided * scale_rescale + noise_pred_uncond.float()

            # Scheduler update in FP32
            out = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
            latents = out.prev_sample

            # Guard against non-finite values
            if not torch.isfinite(latents).all():
                print(f"Non-finite latents at step {i}; sanitizing with nan_to_num.")
                latents = torch.nan_to_num(latents, nan=0.0, posinf=1e3, neginf=-1e3)

            # Track plug-in KL (discrete approximation)
            step_kl = 0.5 * (lambda_k**2) * q_k
            total_kl += step_kl

        if verbose:
            print(
                f"KL: {total_kl:.1f}/{budget:.1f}, "
                f"Lambda: [{min(lambdas):.3f}, {max(lambdas):.3f}]"
            )
            print(
                f"latents stats @end: "
                f"min={latents.min().item():.3f}, "
                f"max={latents.max().item():.3f}, "
                f"std={latents.std().item():.3f}"
            )

        # --- Decode with fp32 VAE ---
        latents_vae = (latents / self.pipe.vae.config.scaling_factor).to(torch.float32)

        image_latents = self.pipe.vae.decode(latents_vae, return_dict=False)[0]
        if verbose:
            print(
                f"decoded stats before clamp: "
                f"min={image_latents.min().item():.3f}, "
                f"max={image_latents.max().item():.3f}, "
                f"std={image_latents.std().item():.3f}"
            )

        # VAE outputs in [-1, 1] – clamp there, then postprocess to PIL
        image_latents = image_latents.clamp(-1, 1)
        image = self.pipe.image_processor.postprocess(image_latents, output_type="pil")[0]

        return image, float(total_kl), lambdas

    @torch.no_grad()
    def generate_fixed(
        self,
        prompt: str,
        guidance_scale: float = 7.5,
        num_steps: int = 40,
        seed: int = 42,
        prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        """
        Standard SDXL generation using the underlying pipeline,
        but decoding via fp32 VAE for stability.

        Parameters
        ----------
        prompt : str
            Main text prompt.
        guidance_scale : float
            CFG scale.
        num_steps : int
            Number of diffusion steps.
        seed : int
            RNG seed.
        prompt_2 : str, optional
            Optional second SDXL text encoder prompt.
        negative_prompt : str, optional
            Negative prompt for the first text encoder.
        negative_prompt_2 : str, optional
            Negative prompt for the second text encoder.
        height, width : int
            Output image size in pixels.

        Returns
        -------
        image : PIL.Image.Image
        """
        generator = torch.Generator(self.device).manual_seed(seed)

        # Ask the pipeline for latents (not PIL)
        out = self.pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="latent",  # Don't let the pipe call fp16 VAE internally
            return_dict=True,
        )

        latents = out.images  # these are latents when output_type="latent"

        # Decode in float32 using fp32 VAE
        latents_vae = (latents / self.pipe.vae.config.scaling_factor).to(torch.float32)
        image_latents = self.pipe.vae.decode(latents_vae, return_dict=False)[0]
        image_latents = image_latents.clamp(-1, 1)  # VAE outputs in [-1, 1]

        image = self.pipe.image_processor.postprocess(image_latents, output_type="pil")[0]
        return image


def make_grid(images: List[Image.Image], labels: List[str]) -> Image.Image:
    """
    Simple horizontal image grid with optional labels drawn above each image.
    """
    if not images:
        raise ValueError("images list must not be empty")

    w, h = images[0].size
    grid = Image.new("RGB", (w * len(images), h + 40))
    draw = ImageDraw.Draw(grid)

    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = i * w
        grid.paste(img, (x_offset, 40))
        if label:
            draw.text((x_offset + 5, 10), label, fill=(255, 255, 255))

    return grid


if __name__ == "__main__":
    # Convenience instance when running as a script/notebook
    ito = ITOPipeline()
