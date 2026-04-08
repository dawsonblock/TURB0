import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path


def _load_preflight_module():
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "turboquant_preflight", repo_root / "scripts" / "preflight.py"
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_runs_from_source_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--json"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout

    payload = json.loads(proc.stdout)
    assert "Cannot import turboquant" not in payload["errors"]
    assert payload["turboquant_version"] is not None
    assert payload["strict_ready"] == (not payload["strict_failures"])
    if payload["strict_failures"]:
        assert payload["strict_ready"] is False


def test_preflight_mlx_version_uses_metadata_fallback(monkeypatch) -> None:
    module = _load_preflight_module()
    fake_mlx = types.ModuleType("mlx")

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.31.1")

    assert module.check_mlx_version() == "0.31.1"
