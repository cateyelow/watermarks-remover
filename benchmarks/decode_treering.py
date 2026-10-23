#!/usr/bin/env python3
"""Tree-Ring watermark decoder for bench_waves.py --decoder interface.

Detects Tree-Ring watermarks by inverting the image back to latent space
via DDIM inversion and checking Fourier-domain correlation with the
known key pattern.

CLI contract (bench_waves.py --decoder):
    python decode_treering.py <image_path> --json
    stdout: {"detected": true/false, "score": <float>}

Requires: torch, diffusers, numpy, Pillow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def circle_mask(size: int, radius: int) -> np.ndarray:
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2


def generate_key_pattern(channels: int, h: int, w: int, radius: int, seed: int) -> np.ndarray:
    """Reproduce the exact Tree-Ring key pattern from seed."""
    rng = np.random.RandomState(seed)
    gt_init = rng.randn(1, channels, h, w).astype(np.float64)
    gt_patch = np.fft.fftshift(np.fft.fft2(gt_init), axes=(-1, -2))

    for i in range(radius, 0, -1):
        mask = circle_mask(min(h, w), i)
        mask_h = mask[:h, :w]
        for j in range(channels):
            val = gt_patch[0, j, 0, i]
            gt_patch[0, j, mask_h] = val

    return gt_patch


def _ddim_inversion(pipe, latent, n_steps: int = 50):
    """Run DDIM inversion to recover initial noise from a final latent.

    Uses an empty prompt (unconditional) for the inversion, which is the
    standard approach when the original prompt is unknown.
    """
    import torch

    pipe.scheduler.set_timesteps(n_steps)

    with torch.no_grad():
        tok = pipe.tokenizer(
            [""],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        prompt_embeds = pipe.text_encoder(tok.input_ids.to(pipe.device))[0]

    inv_timesteps = pipe.scheduler.timesteps.flip(0)
    z = latent.clone()

    with torch.no_grad():
        for i in range(len(inv_timesteps) - 1):
            t = inv_timesteps[i]
            t_next = inv_timesteps[i + 1]

            eps = pipe.unet(z, t, encoder_hidden_states=prompt_embeds).sample

            a_t = pipe.scheduler.alphas_cumprod[t]
            a_next = pipe.scheduler.alphas_cumprod[t_next]

            pred_x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            z = a_next.sqrt() * pred_x0 + (1 - a_next).sqrt() * eps

    return z


def detect_treering(
    image_path: str,
    model_id: str = "huanzi05/stable-diffusion-2-1-base",
    w_radius: int = 10,
    w_seed: int = 999999,
    threshold: float = 0.15,
    n_steps: int = 50,
) -> tuple[bool, float]:
    """Detect Tree-Ring watermark via DDIM inversion + Fourier correlation.

    Steps:
      1. VAE-encode the image to latent space
      2. DDIM-invert the latent back to initial noise (empty prompt)
      3. FFT the recovered noise and compare the ring region against the
         known key pattern using phase coherence

    Returns (detected, score).
    """
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    img = Image.open(image_path).convert("RGB").resize((512, 512))
    img_tensor = (
        torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        latent = pipe.vae.encode(img_tensor).latent_dist.mean
        latent = latent * pipe.vae.config.scaling_factor

    recovered_noise = _ddim_inversion(pipe, latent, n_steps=n_steps)

    noise_np = recovered_noise.cpu().float().numpy()
    _, channels, h, w = noise_np.shape

    noise_fft = np.fft.fftshift(np.fft.fft2(noise_np), axes=(-1, -2))

    key_pattern = generate_key_pattern(channels, h, w, w_radius, w_seed)

    wm_mask = circle_mask(min(h, w), w_radius)[:h, :w]

    scores = []
    for c in range(channels):
        region_noise = noise_fft[0, c][wm_mask]
        region_key = key_pattern[0, c][wm_mask]

        cross = region_noise * region_key.conj()
        unit = cross / np.abs(cross).clip(min=1e-8)
        coherence = np.abs(unit.mean())
        scores.append(float(coherence))

    score = float(np.mean(scores))
    return score > threshold, score


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, help="Image to check")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--model", default="huanzi05/stable-diffusion-2-1-base")
    p.add_argument("--w-radius", type=int, default=10)
    p.add_argument("--w-seed", type=int, default=999999)
    p.add_argument("--threshold", type=float, default=0.15)
    args = p.parse_args()

    if not args.image.is_file():
        print(json.dumps({"error": f"not found: {args.image}"}))
        return 2

    try:
        detected, score = detect_treering(
            str(args.image),
            model_id=args.model,
            w_radius=args.w_radius,
            w_seed=args.w_seed,
            threshold=args.threshold,
        )
    except Exception as e:
        print(json.dumps({"error": str(e), "detected": False}))
        return 1

    if args.json:
        print(json.dumps({"detected": detected, "score": round(score, 4)}))
    else:
        status = "DETECTED" if detected else "clean"
        print(f"{args.image}: {status} (score={score:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
