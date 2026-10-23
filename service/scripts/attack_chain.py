#!/usr/bin/env python3
"""Attack chain coordinator for the WAVES robustness-evaluation toolchain.

Selects and sequences removal engines per watermark type, runs them in order,
and verifies each step with the forensic stealth detector.

Exit codes:
  0  chain completed
  1  runtime error
  2  bad input
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import cleaned_path, emit_json, eprint  # noqa: E402

ATTACK_CHAINS: dict[str, list[dict[str, Any]]] = {
    "tree_ring": [
        {"engine": "clean_treering.py", "args": ["--mode", "filter"]},
        {"engine": "clean_fmdiffwa.py", "args": []},
    ],
    "gaussian_shading": [
        {"engine": "clean_marknull.py", "args": ["--mode", "fast"]},
        {"engine": "clean_fmdiffwa.py", "args": []},
    ],
    "stegastamp": [
        {"engine": "clean_fmdiffwa.py", "args": []},
        {
            "engine": "clean_synthid_bypass.py",
            "args": ["--pipeline", "vae_roundtrip+channel_squeeze"],
        },
    ],
    "stable_signature": [
        {"engine": "clean_nfpa.py", "args": []},
        {"engine": "clean_fmdiffwa.py", "args": []},
    ],
    "synthid": [
        {"engine": "clean_synthid_bypass.py", "args": ["--pipeline", "vae_roundtrip+elastic_warp"]},
        {"engine": "clean_fmdiffwa.py", "args": []},
    ],
    "hidden": [
        {"engine": "clean_fmdiffwa.py", "args": []},
    ],
    "trustmark": [
        {"engine": "clean_fmdiffwa.py", "args": []},
        {"engine": "clean_nfpa.py", "args": []},
    ],
    "unknown": [
        {"engine": "clean_fmdiffwa.py", "args": []},
        {"engine": "clean_marknull.py", "args": ["--mode", "fast"]},
    ],
}

VALID_SCHEMES = frozenset(ATTACK_CHAINS.keys())


class ChainResult(NamedTuple):
    scheme: str
    steps_run: int
    steps_total: int
    stealth_score: float
    elapsed_s: float
    output_path: str


def _run_engine(
    script: str,
    input_path: Path,
    output_path: Path,
    extra_args: list[str],
    device: str | None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / script),
        str(input_path),
        "-o",
        str(output_path),
        "--json",
    ]
    if device:
        cmd.extend(["--device", device])
    cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _run_forensic(image_path: Path) -> float:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "forensic_check.py"), str(image_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode == 0:
        try:
            report = json.loads(result.stdout)
            return float(report.get("stealth_score", 1.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 1.0


def run_chain(
    image_path: Path,
    scheme: str,
    output_path: Path | None = None,
    device: str | None = None,
    timeout: int = 600,
    stealth_threshold: float = 0.7,
) -> ChainResult:
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"unknown scheme: {scheme}; valid: {sorted(VALID_SCHEMES)}")

    chain = ATTACK_CHAINS[scheme]
    t0 = time.monotonic()
    current = image_path
    steps_run = 0

    for i, step in enumerate(chain):
        script = step["engine"]
        if not (SCRIPTS_DIR / script).is_file():
            eprint(f"skipping {script}: not installed")
            continue

        step_out = output_path if (i == len(chain) - 1 and output_path) else None
        if step_out is None:
            step_out = current.parent / f"{current.stem}.step{i}{current.suffix}"

        try:
            result = _run_engine(script, current, step_out, step["args"], device, timeout)
        except subprocess.TimeoutExpired:
            eprint(f"{script} timed out after {timeout}s")
            break

        if result.returncode == 0 and step_out.is_file():
            current = step_out
            steps_run += 1
        elif result.returncode == 3:
            eprint(f"{script} unavailable (exit 3), skipping")
            continue
        else:
            eprint(f"{script} failed (exit {result.returncode}): {result.stderr[:200]}")
            continue

        stealth = _run_forensic(current)
        if stealth < stealth_threshold:
            break

    if output_path and current != output_path:
        import shutil

        shutil.copy2(str(current), str(output_path))
        current = output_path

    final_stealth = _run_forensic(current)
    elapsed = time.monotonic() - t0

    return ChainResult(
        scheme=scheme,
        steps_run=steps_run,
        steps_total=len(chain),
        stealth_score=final_stealth,
        elapsed_s=round(elapsed, 2),
        output_path=str(current),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Input image")
    p.add_argument("-o", "--output", type=Path, help="Output path")
    p.add_argument(
        "--scheme",
        choices=sorted(VALID_SCHEMES),
        default=None,
        help="Watermark scheme (default: auto-classify)",
    )
    p.add_argument("--device", type=str, default=None, help="Device for engines")
    p.add_argument("--timeout", type=int, default=600, help="Per-step timeout (s)")
    p.add_argument(
        "--stealth-threshold",
        type=float,
        default=0.7,
        help="Stop chain when stealth score drops below this (default: 0.7)",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    if not args.path.is_file():
        eprint(f"not a file: {args.path}")
        return 2

    scheme = args.scheme
    if scheme is None:
        try:
            clf_result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "classify_watermark.py"),
                    str(args.path),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            if clf_result.returncode == 0:
                clf_data = json.loads(clf_result.stdout)
                scheme = clf_data.get("scheme", "unknown")
            else:
                scheme = "unknown"
        except Exception:
            scheme = "unknown"

    output = args.output or cleaned_path(args.path, ".chain")

    try:
        result = run_chain(
            args.path,
            scheme,
            output_path=output,
            device=args.device,
            timeout=args.timeout,
            stealth_threshold=args.stealth_threshold,
        )
    except ValueError as e:
        eprint(str(e))
        return 2
    except Exception as e:
        eprint(f"chain error: {e}")
        return 1

    if args.json:
        emit_json(
            {
                "scheme": result.scheme,
                "steps_run": result.steps_run,
                "steps_total": result.steps_total,
                "stealth_score": result.stealth_score,
                "elapsed_s": result.elapsed_s,
                "output": result.output_path,
            }
        )
    else:
        eprint(
            f"chain({result.scheme}): {result.steps_run}/{result.steps_total} steps, "
            f"stealth={result.stealth_score:.3f}, {result.elapsed_s:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
