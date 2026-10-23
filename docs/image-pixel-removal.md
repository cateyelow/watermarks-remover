# Image Pixel-Watermark Removal

Statistical pixel-level watermarks embedded in AI-generated images
(Tree-Ring, Gaussian Shading, StegaStamp, SynthID-Image, etc.) are
invisible but detectable.  This module provides multiple removal engines,
a classifier, an attack-chain coordinator, and forensic stealth
verification.

## Supported Watermark Types

| Category | Schemes |
|----------|---------|
| Latent-space | Tree-Ring, RingID, ROBIN, WIND, SFW, Gaussian Shading, GaussMarker, PRC, SEAL |
| Pixel-domain | StegaStamp, Stable Signature, TrustMark, VINE, HiDDeN |
| Proprietary | SynthID-Image |

## Engines

| Engine | Script | Method |
|--------|--------|--------|
| FLUX 3 regeneration | `clean_flux3.py` | OpenRouter img2img regeneration via FLUX 3 |
| MarkNull | `clean_marknull.py` | NLAS latent-space optimization (SD 2.1) |
| NFPA | `clean_nfpa.py` | Next-frame prediction attack |
| FMDiffWA | `clean_fmdiffwa.py` | Frequency-domain diffusion watermark attack |
| SynthID bypass | `clean_synthid_bypass.py` | Spectral notch filtering for SynthID |
| Tree-Ring attack | `clean_treering.py` | Frequency notch + optional VAE+PGD |
| Re-watermarking | `clean_rewatermark.py` | Injects a competing watermark signal |

## Quick Start

```bash
# Single engine
python service/scripts/clean_image.py input.png output.png --remove-pixel marknull

# Auto mode: classify then chain
python service/scripts/clean_image.py input.png output.png \
    --remove-pixel auto --stealth-check

# Explicit chain with classification
python service/scripts/clean_image.py input.png output.png \
    --remove-pixel chain --classify

# FLUX 3 regeneration (needs OPENROUTER_API_KEY)
python service/scripts/clean_image.py input.png output.png \
    --remove-pixel flux3 --flux3-provider openrouter
```

## Attack Chain (auto mode)

When `--remove-pixel auto` is used (or `--remove-pixel chain --classify`), the system:

1. Runs `classify_watermark.py` to detect the watermark type
2. Looks up the optimal attack chain for that type
3. Executes engines in sequence with quality checks between stages
4. Optionally runs `forensic_check.py` to verify stealth (`--stealth-check`)

Chains are defined in `attack_chain.py` and cover 8 watermark schemes.

## Forensic Stealth

After removal, `forensic_check.py` scores the output on two axes:

- **Spectral deformation**: detects unnatural frequency artifacts
- **JPEG double compression**: detects re-encoding signatures

Outputs with a combined stealth score below the threshold are flagged.

## 4K / Large Image Support

`tile_utils.py` provides `split_tiles`, `merge_tiles`, and `process_tiled`
for processing large images without exhausting GPU memory.  Engines can
call `process_tiled(image, engine_fn)` to automatically split into
overlapping tiles, process each, and merge with boundary-aware cosine
feathering.  This is a utility module; individual engines opt in by
calling it explicitly.

## Evaluation

Evaluation scripts are provided but require the WAVES dataset and a
GPU with torch installed.  No pre-computed results are included.

- **WAVES benchmark** (`benchmarks/bench_waves.py`): per-engine watermark
  removal success rate and image quality metrics.
- **Forensic evaluation** (`benchmarks/bench_forensic.py`): per-engine
  forensic detectability rates.
- **Full matrix** (`benchmarks/run_full_eval.sh`): watermark x engine x
  intensity sweep.  Produces `results/eval_matrix.json`.

To run: download the [WAVES dataset](https://github.com/umd-huang-lab/WAVES),
install torch with CUDA, then:

```bash
./benchmarks/run_full_eval.sh --waves-dir /path/to/WAVES --output-dir results/
```

## Docker

MarkNull and forensic services have dedicated containers in `compose.yaml`:

```bash
docker compose up wr-marknull wr-forensic
```

## Dependencies

Core engines need `numpy`, `scipy`, and `Pillow`.  MarkNull and NFPA
additionally need `torch` and `diffusers` (install separately; not in
`requirements-dev.txt` since they require GPU-specific builds).  FLUX 3
needs `httpx`.  `httpx` and `Pillow` are listed in `requirements-dev.txt`.
