from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_renderer_module():
    script_path = REPO_ROOT / "scripts" / "render_research_docs.py"
    spec = importlib.util.spec_from_file_location(
        "render_research_docs",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_research_docs_match_renderer() -> None:
    renderer = _load_renderer_module()
    rendered = renderer.render_docs()
    for path, expected_content in rendered.items():
        assert path.read_text(encoding="utf-8") == expected_content, (
            f"{path.relative_to(REPO_ROOT)} is out of date; rerun "
            "scripts/render_research_docs.py"
        )


def test_generated_research_docs_preserve_boundaries() -> None:
    benchmark_index = (REPO_ROOT / "docs" / "benchmark_index.md").read_text(
        encoding="utf-8"
    )
    family_matrix = (
        REPO_ROOT / "docs" / "family_evidence_matrix.md"
    ).read_text(encoding="utf-8")

    assert "does not widen the supported product contract" in benchmark_index
    assert "Common Research Report Envelope" in benchmark_index
    assert "tests/integration_research/test_kv_paper_eval_smoke.py" in benchmark_index
    assert "preserves the current asymmetry between Llama and Gemma" in family_matrix
    assert "Gemma does not have the same release-gated paper_mse quality depth as Llama." in family_matrix
