#!/usr/bin/env python3
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.research_catalog import benchmark_surfaces, family_research_adjunct_map
from benchmarks.runtime_cert.research_report_schema import COMMON_RESEARCH_REPORT_FIELDS

CONTRACT_PATH = REPO_ROOT / "turboquant" / "contract.json"
GENERATED_HEADER = (
    "<!-- Generated from benchmarks/research_catalog.py and "
    "turboquant/contract.json by scripts/render_research_docs.py. "
    "Do not edit by hand. -->"
)
JsonDict = dict[str, Any]
FIELD_DESCRIPTIONS = {
    "schema_version": "stable research-report envelope version",
    "metric_family": "benchmark/report family identifier",
    "run_id": "timestamp-derived run label for cross-artifact alignment",
    "timestamp": "captured environment timestamp",
    "preset": "primary preset or preset group for the report",
    "family": "family scope or synthetic/not-applicable scope label",
    "scope": "lane label; research reports stay research-only",
    "mode": "script-level mode identifier",
    "status": "top-level run outcome",
    "metrics": "compact summary metrics for quick comparisons",
    "artifact_paths": "stable companion artifact filenames",
    "notes": "honest interpretation notes retained with the report",
}


def load_contract() -> JsonDict:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return cast(JsonDict, payload)


def _format_code_list(items: list[str]) -> str:
    return "<br>".join(f"`{item}`" for item in items)


def _format_lines(items: list[str]) -> str:
    return "<br>".join(items)


def _family_scope_label(scope: tuple[str, ...]) -> str:
    return ", ".join(f"`{item}`" for item in scope)


def _doc_link(path: str) -> str:
    return f"[{path}]({path})"


def render_benchmark_index(contract: JsonDict) -> str:
    product_surfaces = [item for item in benchmark_surfaces() if item.lane == "product-certification"]
    research_surfaces = [item for item in benchmark_surfaces() if item.lane == "research-only"]
    lines = [
        GENERATED_HEADER,
        "# Benchmark Index",
        "",
        "This generated index maps the repo's retained benchmark and report surfaces to their lane, stable outputs, and proof discipline.",
        "It keeps product certification evidence separate from research-only evidence and does not widen the supported product contract.",
        contract["canonical_runtime"]["source_archive_evidence_rule"],
        "",
        "## Lane Boundary",
        "",
        "- Product-certification surfaces can contribute to release truth only when they travel inside an addressable certification bundle with the required manifest and provenance fields.",
        "- Research-only surfaces emit stable report bundles for comparison and archaeology, but they do not weaken the release gate or promote research into support truth.",
        "",
        "## Product-Certification Surfaces",
        "",
        "| Surface | Primary script | Family scope | Stable outputs | Release role | Primary doc |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for surface in product_surfaces:
        outputs = _format_code_list([artifact.path for artifact in surface.stable_outputs])
        lines.append(
            f"| {surface.title} | `{surface.script}` | {_family_scope_label(surface.family_scope)} | {outputs} | {surface.contract_effect} | {_doc_link(surface.doc_path)} |"
        )

    lines.extend(
        [
            "",
            "The certification workflow also carries runtime smokes, long-context stability, and family-scoped quality guardrails. Those stages are defined in [docs/runtime-certification.md](docs/runtime-certification.md) and remain the product lane's release gate.",
            "",
            "## Research-Only Report Surfaces",
            "",
            "| Surface | Metric family | Primary script | Stable outputs | Deterministic smoke | Scope note | Primary doc |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for surface in research_surfaces:
        outputs = _format_code_list([artifact.path for artifact in surface.stable_outputs])
        smoke = f"`{surface.smoke_test}`" if surface.smoke_test else "-"
        lines.append(
            f"| {surface.title} | `{surface.metric_family}` | `{surface.script}` | {outputs} | {smoke} | {surface.contract_effect} | {_doc_link(surface.doc_path)} |"
        )

    lines.extend(
        [
            "",
            "## Common Research Report Envelope",
            "",
            "The research-only report writers share one stable top-level envelope from `benchmarks/runtime_cert/research_report_schema.py` so JSON payloads stay comparable across commits.",
            "",
            "| Field | Meaning |",
            "| :--- | :--- |",
        ]
    )
    for field in COMMON_RESEARCH_REPORT_FIELDS:
        lines.append(f"| `{field}` | {FIELD_DESCRIPTIONS[field]} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `inner_product_bias` and `bit_budget_sweep` are synthetic by design and remain outside family-level runtime proof.",
            "- `kv_paper_eval` explicitly distinguishes `fast-check` from `heavy-offline` stages so missing heavy evidence is never silently fabricated.",
            "- `vector_search` remains a research lane even when its summary is carried alongside a dated benchmark snapshot.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_family_evidence_matrix(contract: JsonDict) -> str:
    adjuncts = family_research_adjunct_map()
    lines = [
        GENERATED_HEADER,
        "# Family Evidence Matrix",
        "",
        "This generated matrix separates release-gated family evidence from research-only adjunct evidence.",
        "It preserves the current asymmetry between Llama and Gemma instead of flattening them into one support depth.",
        contract["canonical_runtime"]["source_archive_evidence_rule"],
        "",
        "| Family | Product evidence depth | Release-gated evidence | Research-only adjunct evidence | Still unproven |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for family in contract["families"]:
        name = str(family["name"])
        adjunct = adjuncts[name]
        release_evidence = _format_lines([str(item) for item in family["coverage"]])
        research_only = _format_lines(list(adjunct.research_only_evidence))
        still_unproven = _format_lines([str(family["notes"]), *adjunct.still_unproven])
        lines.append(
            f"| {family['display_name']} | {family['evidence_depth']} | {release_evidence} | {research_only} | {still_unproven} |"
        )

    family_agnostic = [
        surface
        for surface in benchmark_surfaces()
        if surface.lane == "research-only"
        and set(surface.family_scope).issubset({"synthetic", "not-applicable"})
    ]
    lines.extend(
        [
            "",
            "## Family-Agnostic Research Lanes",
            "",
            "| Surface | Metric family | Family scope | Why it does not change family evidence depth |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )
    for surface in family_agnostic:
        lines.append(
            f"| {surface.title} | `{surface.metric_family}` | {_family_scope_label(surface.family_scope)} | {surface.contract_effect} |"
        )

    lines.extend(
        [
            "",
            "## Current Asymmetry",
            "",
            "- Llama retains the stronger release-gated evidence depth in the current contract.",
            "- Gemma remains narrower because the conservative paper_mse quality guardrail is not symmetric with Llama.",
            "- Synthetic and retrieval research lanes can inform future work, but they do not by themselves produce runtime-complete family proof.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_docs() -> dict[Path, str]:
    contract = load_contract()
    return {
        REPO_ROOT / "docs" / "benchmark_index.md": render_benchmark_index(contract),
        REPO_ROOT / "docs" / "family_evidence_matrix.md": render_family_evidence_matrix(contract),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render research-lane index docs")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if rendered research docs differ from checked-in docs",
    )
    args = parser.parse_args()

    mismatches: list[str] = []
    for path, content in render_docs().items():
        if args.check:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            if existing != content:
                mismatches.append(str(path.relative_to(REPO_ROOT)))
        else:
            path.write_text(content, encoding="utf-8")

    if mismatches:
        raise SystemExit(
            "Rendered research docs are out of date: " + ", ".join(mismatches)
        )


if __name__ == "__main__":
    main()
