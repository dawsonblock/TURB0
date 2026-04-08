import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _load_audit_module():
    spec = spec_from_file_location(
        "turboquant_audit_vendored_surface",
        REPO_ROOT / "tools" / "audit_vendored_surface.py",
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boundary_audit_passes_and_tracks_live_surface() -> None:
    audit_module = _load_audit_module()
    proc = subprocess.run(
        [sys.executable, "tools/audit_vendored_surface.py", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["canonical_entry_exists"] is True
    assert payload["declared_repo_paths"] == list(audit_module.REQUIRED_REPO_PATHS)
    assert payload["forbidden_repo_paths"] == []
    assert payload["missing_repo_paths"] == []
    assert payload["missing_hooks"] == []
    assert payload["mismatched_hooks"] is False


def test_active_tooling_does_not_reference_in_tree_mlx_lm_directory() -> None:
    audit_tool = _read("tools/audit_vendored_surface.py")
    assert "MLX_LM_DIR" not in audit_tool
    assert "scan_mlx_lm" not in audit_tool
    assert 'REPO_ROOT / "mlx_lm"' not in audit_tool

    for rel_path in (
        ".github/workflows/apple-runtime-cert.yml",
        "scripts/preflight.py",
        "scripts/validate_local.sh",
    ):
        content = _read(rel_path)
        assert "mlx_lm/" not in content, (
            f"{rel_path} must not reference an in-tree mlx_lm/ repo path."
        )


def test_release_docs_require_retained_runtime_cert_artifacts() -> None:
    content = _read("docs/release-checklist.md")
    normalized = " ".join(content.lower().split())

    assert "release candidate is not called certified" in normalized
    assert "retained `runtime-cert-<sha>` artifact bundle" in content
    assert "exact commit or tag" in normalized
