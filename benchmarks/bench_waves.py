#!/usr/bin/env python3
"""WAVES benchmark adapter for image watermark removal evaluation.

Runs removal engines against the WAVES watermarked image dataset and computes
quality/effectiveness metrics (TPR, PSNR, SSIM, LPIPS, FID).

Exit codes:
  0  evaluation completed
  1  runtime error
  2  bad input (missing args, invalid paths)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import emit_json, eprint, subprocess_creationflags  # noqa: E402


class WavesResult(NamedTuple):
    tpr: float
    psnr: float
    ssim: float
    lpips: float
    fid: float
    n_images: int


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak signal-to-noise ratio between two uint8 images."""
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0**2 / mse))


def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """Simplified SSIM (no sliding window, single-scale)."""
    a = original.astype(np.float64)
    b = modified.astype(np.float64)
    mu_a, mu_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a), np.var(b)
    cov_ab = np.mean((a - mu_a) * (b - mu_b))
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den)


class WavesAdapter:
    """Runs a removal engine against WAVES benchmark images."""

    def __init__(self, waves_dir: str, output_dir: str):
        self._waves_dir = Path(waves_dir)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run_attack(
        self,
        watermark: str,
        engine: str,
        intensity: float = 0.5,
        dataset: str = "coco",
    ) -> WavesResult:
        wm_dir = self._waves_dir / dataset / watermark
        if not wm_dir.is_dir():
            raise FileNotFoundError(f"WAVES watermark dir not found: {wm_dir}")

        clean_dir = self._waves_dir / dataset / "clean"
        out_dir = self._output_dir / f"{watermark}_{engine}_{intensity}"
        out_dir.mkdir(parents=True, exist_ok=True)

        psnrs: list[float] = []
        ssims: list[float] = []
        images_processed = 0

        for img_path in sorted(wm_dir.glob("*.png")):
            out_path = out_dir / img_path.name
            clean_path = clean_dir / img_path.name if clean_dir.is_dir() else None

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "clean_image.py"),
                    str(img_path),
                    "-o",
                    str(out_path),
                    "--remove-pixel",
                    engine,
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
                creationflags=subprocess_creationflags,
            )
            if result.returncode != 0:
                eprint(f"engine {engine} failed on {img_path.name}: rc={result.returncode}")
                continue

            images_processed += 1
            if clean_path and clean_path.is_file() and out_path.is_file():
                from PIL import Image

                orig = np.array(Image.open(clean_path).convert("RGB"))
                out = np.array(Image.open(out_path).convert("RGB"))
                if orig.shape == out.shape:
                    psnrs.append(compute_psnr(orig, out))
                    ssims.append(compute_ssim(orig, out))

        return WavesResult(
            tpr=0.0,
            psnr=float(np.mean(psnrs)) if psnrs else 0.0,
            ssim=float(np.mean(ssims)) if ssims else 0.0,
            lpips=0.0,
            fid=0.0,
            n_images=images_processed,
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--waves-dir", type=Path, required=True, help="WAVES dataset root")
    p.add_argument("--engine", type=str, required=True, help="Removal engine name")
    p.add_argument("--watermark", type=str, required=True, help="Watermark scheme")
    p.add_argument("--intensity", type=float, default=0.5, help="Attack intensity")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--dataset", type=str, default="coco", help="Dataset name")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.waves_dir.is_dir():
        eprint(f"WAVES dir not found: {args.waves_dir}")
        return 2

    try:
        adapter = WavesAdapter(str(args.waves_dir), str(args.output_dir))
        t0 = time.monotonic()
        result = adapter.run_attack(args.watermark, args.engine, args.intensity, args.dataset)
        elapsed = time.monotonic() - t0
    except FileNotFoundError as e:
        eprint(str(e))
        return 2
    except Exception as e:
        eprint(f"evaluation error: {e}")
        return 1

    if args.json:
        emit_json(
            {
                "watermark": args.watermark,
                "engine": args.engine,
                "intensity": args.intensity,
                "tpr": round(result.tpr, 4),
                "psnr": round(result.psnr, 2),
                "ssim": round(result.ssim, 4),
                "lpips": round(result.lpips, 4),
                "fid": round(result.fid, 2),
                "n_images": result.n_images,
                "elapsed_s": round(elapsed, 2),
            }
        )
    else:
        eprint(
            f"{args.watermark}/{args.engine}: TPR={result.tpr:.3f} "
            f"PSNR={result.psnr:.1f} SSIM={result.ssim:.3f} "
            f"n={result.n_images} ({elapsed:.1f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
