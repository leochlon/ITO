#!/usr/bin/env python3
"""
General-purpose ITO pipeline script for image generation.

This script demonstrates:
- ITO-guided image generation with adaptive guidance
- Baseline SDXL generation with fixed CFG
- Side-by-side comparison of both approaches

Usage:
    source .venv/bin/activate
    python generate.py --prompt "your prompt here"

Examples:
    python generate.py --prompt "a cozy cabin in a snowy forest"
    python generate.py --prompt "a futuristic city" --steps 50 --seed 123
    python generate.py --prompt "a serene lake" --output lake --budget 50
"""

import argparse
from ITO import ITOPipeline, make_grid

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate images using ITO pipeline with adaptive guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="a cozy cabin in a snowy forest, warm lights in the windows, winter evening, cinematic lighting, highly detailed",
        help="Main prompt for image generation"
    )
    parser.add_argument(
        "--negative", "-n",
        type=str,
        default="low quality, blurry, distorted, text, watermark, deformed",
        help="Negative prompt"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output filename prefix (without extension)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=40.0,
        help="KL divergence budget for ITO"
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=7.5,
        help="Maximum guidance scale"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Alpha parameter for ITO scheduling"
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.5,
        help="Fixed CFG value for baseline comparison"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip baseline generation and comparison grid"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ITO Pipeline - Image Generation")
    print("=" * 60)
    print()

    # Initialize the pipeline
    print("Loading SDXL pipeline...")
    print("(First run will download the model - this may take a while)")
    ito = ITOPipeline()
    print("Pipeline loaded successfully!")
    print()

    # Use parsed arguments
    prompt = args.prompt
    negative_prompt = args.negative
    seed = args.seed
    num_steps = args.steps
    height = args.height
    width = args.width

    print("Prompt:", prompt)
    print("Negative:", negative_prompt)
    print(f"Settings: {num_steps} steps, {height}x{width}px, seed={seed}")
    print()

    # Generate with ITO guidance
    print("-" * 60)
    print("Generating with ITO guidance...")
    print("-" * 60)
    image_ito, total_kl, lambdas = ito.generate_ito(
        prompt=prompt,
        negative_prompt=negative_prompt,
        budget=args.budget,
        lambda_max=args.lambda_max,
        num_steps=num_steps,
        height=height,
        width=width,
        alpha=args.alpha,
        seed=seed,
        verbose=True,
    )
    ito_filename = f"{args.output}_ito.png"
    image_ito.save(ito_filename)
    print(f"ITO image saved to: {ito_filename}")
    print(f"Total KL used: {total_kl:.2f}")
    print(f"Lambda range: [{min(lambdas):.2f}, {max(lambdas):.2f}]")
    print()

    generated_files = [f"  - {ito_filename} (ITO-guided)"]

    if not args.no_comparison:
        # Generate with fixed CFG (baseline)
        print("-" * 60)
        print("Generating with fixed CFG (baseline)...")
        print("-" * 60)
        image_fixed = ito.generate_fixed(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=args.cfg,
            num_steps=num_steps,
            height=height,
            width=width,
            seed=seed,
        )
        fixed_filename = f"{args.output}_fixed.png"
        image_fixed.save(fixed_filename)
        print(f"Baseline image saved to: {fixed_filename}")
        print()

        # Create comparison grid
        print("-" * 60)
        print("Creating comparison grid...")
        print("-" * 60)
        grid = make_grid(
            images=[image_ito, image_fixed],
            labels=[f"ITO (KL={total_kl:.1f})", f"Fixed CFG={args.cfg}"],
        )
        comparison_filename = f"{args.output}_comparison.png"
        grid.save(comparison_filename)
        print(f"Comparison grid saved to: {comparison_filename}")
        print()

        generated_files.extend([
            f"  - {fixed_filename} (baseline)",
            f"  - {comparison_filename} (side-by-side)"
        ])

    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    for file in generated_files:
        print(file)
    print()

if __name__ == "__main__":
    main()
