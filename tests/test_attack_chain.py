"""Tests for the attack chain coordinator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

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


def test_json_output(tmp_path, monkeypatch):
    import json

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    img = tmp_path / "test.png"
    _write_random_png(img, 64, 64)
    result = _run("attack_chain.py", str(img), "--scheme", "unknown", "--json")
    if result.returncode == 0:
        report = json.loads(result.stdout)
        assert "scheme" in report
        assert "steps_run" in report
        assert "stealth_score" in report


def test_auto_classify_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    img = tmp_path / "test.png"
    _write_random_png(img, 64, 64)
    result = _run("attack_chain.py", str(img), "--json")
    assert result.returncode in (0, 1)


def test_flux3_step_is_reached_after_fft(tmp_path):
    """Verify that the Flux3 step is NOT skipped by the forensic early stop."""
    import shutil as _shutil

    from attack_chain import run_chain

    img = tmp_path / "input.png"
    _write_random_png(img, 128, 128)
    out = tmp_path / "output.png"

    engines_called: list[str] = []

    def mock_run(cmd, **kwargs):
        script_name = Path(cmd[1]).name if len(cmd) > 1 else ""
        engines_called.append(script_name)
        out_idx = cmd.index("-o") + 1 if "-o" in cmd else -1
        step_out = Path(cmd[out_idx]) if out_idx > 0 else out
        _shutil.copy2(str(img), str(step_out))
        if script_name == "forensic_check.py":
            return subprocess.CompletedProcess(cmd, 0, stdout='{"stealth_score":0.05}', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}', stderr="")

    with patch("attack_chain.subprocess.run", side_effect=mock_run):
        result = run_chain(img, "unknown", output_path=out, stealth_threshold=0.7)

    assert result.steps_run == 2, f"expected both steps to run, got {result.steps_run}"
    assert "clean_flux3.py" in engines_called, f"Flux3 was never called: {engines_called}"


def test_pick_flux3_resolution_selects_cost_effective_tier():
    from attack_chain import _pick_flux3_resolution

    assert _pick_flux3_resolution(2304, 1728) == "2k"
    assert _pick_flux3_resolution(480, 480) == "768"
    assert _pick_flux3_resolution(1024, 1536) == "1.5k"
    assert _pick_flux3_resolution(4096, 4096) == "4k"


def test_resolve_api_key_from_env():
    from attack_chain import _resolve_api_key

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"}):
        assert _resolve_api_key(None) == "test-key-123"
    assert _resolve_api_key("explicit-key") == "explicit-key"


def test_resolve_api_key_unreadable_file(tmp_path):
    from attack_chain import _resolve_api_key

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False), \
         patch("attack_chain.Path.home", return_value=tmp_path):
            env_file = tmp_path / ".claude" / "openrouter.env"
            env_file.parent.mkdir(parents=True, exist_ok=True)
            env_file.write_text("OPENROUTER_API_KEY=file-key")
            env_file.chmod(0o000)
            key = _resolve_api_key(None)
            env_file.chmod(0o644)
            assert key in ("", "file-key")


def test_api_key_not_in_process_args(tmp_path):
    """Verify API key is passed via env, not as a CLI argument."""
    from attack_chain import _FLUX3_ENGINES, run_chain

    img = tmp_path / "input.png"
    _write_random_png(img, 64, 64)
    out = tmp_path / "output.png"

    captured_cmds: list[list[str]] = []
    captured_envs: list[dict | None] = []

    def mock_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        captured_envs.append(kwargs.get("env"))
        script_name = Path(cmd[1]).name if len(cmd) > 1 else ""
        if script_name in _FLUX3_ENGINES:
            import shutil

            shutil.copy2(str(img), str(out))
            return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}', stderr="")
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="unavailable")

    with patch("attack_chain.subprocess.run", side_effect=mock_run):
        run_chain(img, "unknown", output_path=out, api_key="secret-key-123")

    for cmd in captured_cmds:
        assert "secret-key-123" not in cmd, "API key must not appear in process arguments"

    flux3_envs = [e for cmd, e in zip(captured_cmds, captured_envs, strict=True)
                  if Path(cmd[1]).name in _FLUX3_ENGINES]
    assert any(e and e.get("OPENROUTER_API_KEY") == "secret-key-123" for e in flux3_envs), \
        "API key must be passed via OPENROUTER_API_KEY env var"
