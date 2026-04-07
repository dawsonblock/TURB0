from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

from turboquant.runtime.support import SUPPORTED_FAMILIES

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "turboquant" / "contract.json"
JsonDict = dict[str, Any]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _load_contract() -> JsonDict:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return cast(JsonDict, payload)


def _load_renderer_module():
    script_path = REPO_ROOT / "scripts" / "render_support_contract.py"
    spec = importlib.util.spec_from_file_location(
        "render_support_contract",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_contract_docs_match_renderer() -> None:
    """Generated contract docs must match the renderer output exactly."""
    renderer = _load_renderer_module()
    rendered = renderer.render_docs()
    for path, expected_content in rendered.items():
        assert path.read_text(encoding="utf-8") == expected_content, (
            f"{path.relative_to(REPO_ROOT)} is out of date; rerun "
            "scripts/render_support_contract.py"
        )


def test_allowlisted_families_match_contract_json() -> None:
    """Runtime allowlist must match the machine-readable support contract."""
    contract = _load_contract()
    expected = {
        family["name"]
        for family in contract["families"]
        if family["status"] == "allowlisted"
    }
    assert set(SUPPORTED_FAMILIES) == expected == {"llama", "gemma"}


def test_readme_tracks_primary_contract_story() -> None:
    """README must reflect the runtime contract and its boundaries."""
    content = _read("README.md")
    lowered = content.lower()
    normalized = " ".join(lowered.split())

    assert "Contract summary:" in content
    assert "upgrade_cache_list(...)" in content
    assert "TurboQuantKCache(...)" in content
    assert "KVCache.to_turboquant()" in content
    assert "does not automatically persist `events.jsonl`" in content
    assert "TinyModel" in content
    assert "schema_version == 4" in content
    assert "paper_prod" in content and "paper_mse" in content
    assert "docs/preset_modes.md" in content
    assert "docs/bit_budget_sweep.md" in content
    assert "docs/kv_paper_eval.md" in content
    assert "legacy_topk" in content
    assert "compatibility-only" in lowered or "compatibility only" in lowered
    assert "artifacts/runtime-cert/" in content
    assert "built wheels and source distributions do not ship" in lowered
    assert "apple silicon is required for runtime inference" in lowered
    assert "not a runtime go/no-go" in normalized
    assert "do not, by themselves, prove a current apple runtime pass" in normalized
    assert "only a published certification artifact or pinned manifest digest" in normalized
    assert "encode_topk_residual" not in content
    assert "top-k sparse residual" not in lowered


def test_release_facing_docs_use_addressable_evidence_language() -> None:
    """Release-facing docs must use addressable evidence language."""
    for rel_path in (
        "RELEASE_CANDIDATE_NOTES.md",
        "README.md",
        "docs/contract_status.md",
        "docs/contract_audit.md",
        "docs/product_contract.md",
        "docs/supported-surface.md",
        "docs/support_matrix.md",
        "docs/runtime-certification.md",
        "docs/release-checklist.md",
    ):
        content = _read(rel_path)
        lowered = content.lower()
        assert "addressable" in lowered or "manifest digest" in lowered, (
            f"{rel_path} must use addressable-evidence language."
        )
        assert "artifact-backed" not in lowered, (
            f"{rel_path} must not use the stale 'artifact-backed' wording."
        )
        assert "retained local evidence in this checkout" not in lowered, (
            f"{rel_path} must not imply timestamped local evidence ships with the snapshot."
        )
        assert "addressable from this workspace" not in lowered, (
            f"{rel_path} must not imply portable source snapshots carry local evidence directories."
        )


def test_runtime_certification_doc_tracks_scope_and_artifacts() -> None:
    """runtime-certification.md must describe the evidence contract."""
    content = _read("docs/runtime-certification.md")
    lowered = content.lower()

    assert (
        "artifacts/runtime-cert/<timestamp>/" in content
        or "artifacts/runtime-cert/<timestamp>" in content
    )
    assert "contract.json" in content
    assert "certification_scope.families" in content
    assert "events.jsonl" in content and "optional" in lowered
    assert "teacher-forcing" in lowered or "batch" in lowered
    assert "python3.11" in content
    assert "python3.10" in content
    assert "python3.9" in content
    assert "llama" in lowered and "gemma" in lowered
    assert "stronger" in lowered and "narrower" in lowered
    assert "built wheels and source distributions do not ship" in lowered


def test_evaluation_and_benchmark_docs_are_non_certification_guides() -> None:
    """Exploratory docs must not present heuristics as certification claims."""
    eval_content = _read("docs/evaluation.md").lower()
    bench_content = _read("docs/benchmark_methodology.md").lower()

    assert "not certification gates" in eval_content
    assert "heuristic" in eval_content
    assert "not part of the certified product contract" in bench_content
    for field in (
        "artifact_uri_or_manifest_digest",
        "git_commit",
        "model_ids",
        "mlx_version",
        "hardware",
        "script",
        "args",
    ):
        assert field in bench_content


def test_integration_doc_no_blanket_support_claim() -> None:
    """integration.md must not imply that base.py dispatch grants support."""
    content = _read("docs/integration.md")
    lowered = content.lower()

    assert "will automatically support turboquant" not in lowered
    assert "works out of the box" not in lowered
    assert "model_family" in content
    assert (
        "Routing through `base.py` is not the same as being in the "
        "supported allowlist." in content
    )


def test_supported_surface_generated_doc_has_secondary_surfaces() -> None:
    """supported-surface.md must distinguish canonical and secondary paths."""
    content = _read("docs/supported-surface.md")

    assert "upgrade_cache_list" in content
    assert "Secondary surfaces" in content
    assert "turboquant.integrations.mlx._cache_adapter.TurboQuantKCache" in content
    assert "turboquant.integrations.mlx.cache_adapter.TurboQuantKCache" in content
    assert "KVCache.to_turboquant()" in content
    assert "contract.json" in content
    assert (
        "| `turboquant.integrations.mlx.cache_adapter.TurboQuantKCache` | "
        "compatibility shim | bypasses the model-family allowlist | "
        "`turboquant.integrations.mlx.upgrade.upgrade_cache_list` |" in content
    ), (
        "The compatibility shim must still point callers back to the "
        "canonical support-gated upgrade_cache_list(...) path."
    )


def test_runtime_api_points_to_upgrade_cache_list() -> None:
    """runtime/api.py must still point callers at the canonical path."""
    content = _read("turboquant/runtime/api.py")
    assert "upgrade_cache_list" in content
    assert "Do not instantiate TurboQuantKCache directly" in content


def test_vendored_doc_marks_to_turboquant_as_secondary_helper() -> None:
    """VENDORED_MLX_LM.md must classify to_turboquant() as secondary."""
    content = _read("VENDORED_MLX_LM.md")
    lowered = content.lower()

    assert "to_turboquant()" in content
    assert "deprecated" in lowered
    assert "bypasses the turboquant model-family support gate" in lowered
    assert "canonical public path is `upgrade_cache_list(...)`" in content


def test_validation_local_distinguishes_smoke_from_real_certification() -> None:
    """validation-local.md must distinguish TinyModel smoke from real-model
    certification.
    """
    content = _read("docs/validation-local.md")
    lowered = content.lower()

    assert "TinyModel" in content
    assert (
        "full real-model certification" in lowered
        or "full runtime certification" in lowered
    )
    assert "buildable" in lowered
    assert "statically coherent" in lowered
    assert "not runtime-proven on target hardware" in lowered


def test_contract_status_distinguishes_static_from_runtime_proof() -> None:
    """contract_status.md must not present generic validation as runtime
    proof.
    """
    content = _read("docs/contract_status.md")
    lowered = content.lower()
    normalized = " ".join(lowered.split())

    assert "buildable" in lowered
    assert "statically coherent" in lowered
    assert "not runtime-proven on target" in normalized
    assert "support-contract" in lowered
    assert "typecheck" in lowered
    assert "and those lanes pass" in normalized
    assert "published workflow artifact" in lowered or "manifest digest" in lowered
    assert "retained local evidence in this checkout" not in lowered
