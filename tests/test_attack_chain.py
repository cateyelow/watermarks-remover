"""Tests for the attack chain coordinator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _write_random_png(path: Path, w: int, h: int) -> None:
    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    img.save(path)


def test_valid_schemes_cover_all_known():
    from attack_chain import VALID_SCHEMES

    expected = {
        "tree_ring",
        "gaussian_shading",
        "stegastamp",
        "stable_signature",
        "synthid",
        "hidden",
        "trustmark",
        "unknown",
    }
    assert expected == VALID_SCHEMES


def test_chain_result_structure():
    from attack_chain import ChainResult

    r = ChainResult(
        scheme="unknown",
        steps_run=1,
        steps_total=2,
        stealth_score=0.15,
        elapsed_s=1.5,
        output_path="out.png",
    )
    assert r.scheme == "unknown"
    assert r.steps_run <= r.steps_total


def test_bad_input_exits_2():
    result = _run("attack_chain.py", "/nonexistent.png")
    assert result.returncode == 2


def test_invalid_scheme_exits_2(tmp_path):
    img = tmp_path / "test.png"
    _write_random_png(img, 64, 64)
    result = _run("attack_chain.py", str(img), "--scheme", "nonexistent")
    assert result.returncode == 2


def test_json_output(tmp_path):
    import json

    img = tmp_path / "test.png"
    _write_random_png(img, 64, 64)
    result = _run("attack_chain.py", str(img), "--scheme", "unknown", "--json")
    if result.returncode == 0:
        report = json.loads(result.stdout)
        assert "scheme" in report
        assert "steps_run" in report
        assert "stealth_score" in report


def test_auto_classify_fallback(tmp_path):
    img = tmp_path / "test.png"
    _write_random_png(img, 64, 64)
    result = _run("attack_chain.py", str(img), "--json")
    assert result.returncode in (0, 1)
