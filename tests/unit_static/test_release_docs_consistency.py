import json
import re
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "turboquant" / "contract.json"


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _load_contract() -> dict[str, object]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return cast(dict[str, object], payload)


def test_release_checklist_uses_integration_mlx() -> None:
    text = _read("docs/release-checklist.md")

    assert "tests/integration_mlx" in text
    assert "pytest tests/integration/" not in text


def test_contract_status_tracks_current_supported_boundary() -> None:
    text = _read("docs/contract_status.md")
    lowered = text.lower()

    assert "polarquant_exp" in text
    assert "supported non-paper-facing" in lowered
    assert "outside the supported contract" not in lowered
    assert "artifacts/runtime-cert/" in text
    assert "vendored `mlx_lm`" in text or "vendored mlx_lm" in lowered
    assert "built wheels and source distributions do not ship" in lowered
    assert "published workflow artifact" in lowered or "manifest digest" in lowered
    assert "retained local evidence in this checkout" not in lowered
    assert "source snapshot" in lowered or "source archive" in lowered


def test_release_docs_distinguish_checkout_from_built_distributions() -> None:
    for rel_path in (
        "README.md",
        "docs/contract_status.md",
        "docs/runtime-certification.md",
        "docs/release-checklist.md",
    ):
        content = _read(rel_path)
        normalized = " ".join(content.lower().split())
        assert "artifacts/runtime-cert/" in content, (
            f"{rel_path} must mention the runtime-cert artifact path."
        )
        assert "built wheels and source distributions do not ship" in normalized, (
            f"{rel_path} must distinguish a working tree from shipped distributions."
        )
        assert (
            "published workflow artifact" in normalized
            or "workflow-uploaded artifact" in normalized
            or "manifest digest" in normalized
        ), f"{rel_path} must describe the required external release evidence."


def test_release_facing_docs_do_not_inventory_timestamped_runtime_artifacts() -> None:
    timestamped_runtime_artifacts = re.compile(r"artifacts/runtime-cert/20\d{6,}")

    for rel_path in (
        "RELEASE_CANDIDATE_NOTES.md",
        "docs/contract_status.md",
        "docs/contract_audit.md",
    ):
        content = _read(rel_path)
        assert not timestamped_runtime_artifacts.search(content), (
            f"{rel_path} must not inventory timestamped local runtime artifacts."
        )


def test_release_candidate_notes_distinguish_structure_from_release_proof() -> None:
    content = _read("RELEASE_CANDIDATE_NOTES.md")
    normalized = " ".join(content.lower().split())

    assert "does not, by itself, prove a current apple runtime pass" in normalized
    assert "manifest digest" in normalized
    assert "self-hosted" in normalized


def test_release_checklist_matches_contract_required_artifacts() -> None:
    contract = _load_contract()
    checklist = _read("docs/release-checklist.md")

    runtime = cast(dict[str, Any], contract["canonical_runtime"])
    required = cast(list[str], runtime["required_release_artifacts"])
    for artifact in required:
        assert f"`{artifact}`" in checklist, (
            "docs/release-checklist.md must enumerate every contract-driven "
            f"release artifact, missing {artifact!r}."
        )
