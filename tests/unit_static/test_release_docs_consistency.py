from pathlib import Path


def test_release_checklist_uses_integration_mlx() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "docs" / "release-checklist.md").read_text(
        encoding="utf-8"
    )

    assert "tests/integration_mlx" in text
    assert "pytest tests/integration/" not in text


def test_contract_status_tracks_current_supported_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "docs" / "contract_status.md").read_text(
        encoding="utf-8"
    )
    lowered = text.lower()

    assert "polarquant_exp" in text
    assert "supported non-paper-facing" in lowered
    assert "outside the supported contract" not in lowered
    assert "artifacts/runtime-cert/" in text
    assert "vendored `mlx_lm`" in text or "vendored mlx_lm" in lowered
