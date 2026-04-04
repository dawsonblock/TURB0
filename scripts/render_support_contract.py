from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "turboquant" / "contract.json"
GENERATED_HEADER = (
    "<!-- Generated from turboquant/contract.json by "
    "scripts/render_support_contract.py. Do not edit by hand. -->"
)


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _k_bpc_d128(preset: dict) -> str:
    residual_kind = preset["residual_kind"]
    k_bits = int(preset["k_bits"])
    k_group_size = int(preset["k_group_size"])
    if residual_kind == "qjl":
        qjl_dim = int(preset["qjl_dim"])
        value = (k_bits - 1) + 16.0 / k_group_size + (qjl_dim + 16.0) / 128.0
    elif residual_kind == "none":
        value = k_bits + 16.0 / k_group_size
    else:
        return "legacy / compatibility-only"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _avg_kv_bpc_d128(preset: dict) -> str:
    residual_kind = preset["residual_kind"]
    if residual_kind == "topk":
        return "legacy / compatibility-only"

    k_bits = int(preset["k_bits"])
    k_group_size = int(preset["k_group_size"])
    v_bits = int(preset["v_bits"])
    v_group_size = int(preset["v_group_size"])
    if residual_kind == "qjl":
        qjl_dim = int(preset["qjl_dim"])
        k_bpc = (k_bits - 1) + 16.0 / k_group_size + (qjl_dim + 16.0) / 128.0
    else:
        k_bpc = k_bits + 16.0 / k_group_size
    v_bpc = v_bits + 16.0 / v_group_size
    value = (k_bpc + v_bpc) / 2.0
    return f"{value:.3f}".rstrip("0").rstrip(".")


def render_support_matrix(contract: dict) -> str:
    lines = [
        GENERATED_HEADER,
        "# TurboQuant Support Matrix",
        "",
        "TurboQuant's narrow support boundary is generated from "
        "`turboquant/contract.json`.",
        "A source archive alone does not prove a current PASS run; use an "
        "addressable workflow artifact, release evidence bundle, or pinned "
        "manifest digest when making evidence claims.",
        "",
        "## Algorithm Presets",
        "",
        "| Preset | Canonical algorithm | Residual | Effective K bpc (d=128) | Average KV bpc (d=128) | Notes |",
        "| :--- | :--- | :--- | :---: | :---: | :--- |",
    ]
    for preset in contract["presets"]:
        lines.append(
            f"| `{preset['display_name']}` | `{preset['algorithm']}` | "
            f"`{preset['residual_kind']}` | {_k_bpc_d128(preset)} | "
            f"{_avg_kv_bpc_d128(preset)} | {preset['notes']} |"
        )

    lines.extend(
        [
            "",
            "Paper-facing presets are `paper_mse` and `paper_prod` (the "
            "`paper_prod_qjl` algorithm family). Legacy top-k presets remain "
            "available only as compatibility surfaces.",
            "",
            "## Exact deviations from the paper-facing story",
            "",
        ]
    )
    for deviation in contract["deviations"]:
        lines.append(f"- **{deviation['title']}** — {deviation['description']}")

    lines.extend(
        [
            "",
            "## Model Architecture Matrix",
            "",
            "| Model family | Canonical support status | Evidence depth | Notes |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )
    for family in contract["families"]:
        lines.append(
            f"| {family['display_name']} | Allowlisted via `upgrade_cache_list(...)` | "
            f"{family['evidence_depth']} | {family['workflow_story']} "
            f"{family['notes']} Coverage: {', '.join(family['coverage'])}. |"
        )

    lines.extend(
        [
            "",
            "## MLX Compatibility",
            "",
            f"- MLX >= {contract['canonical_runtime']['mlx']['min']} "
            f"and < {contract['canonical_runtime']['mlx']['max_exclusive']}",
            "",
            "## Hardware",
            "",
            "- Apple Silicon (darwin-arm64)",
        ]
    )
    return "\n".join(lines) + "\n"


def render_supported_surface(contract: dict) -> str:
    runtime = contract["canonical_runtime"]
    lines = [
        GENERATED_HEADER,
        "# Supported surface",
        "",
        "TurboQuant's supported surface is generated from "
        "`turboquant/contract.json`.",
        runtime["source_archive_evidence_rule"],
        "",
        "## Supported slice",
        "",
        f"- {runtime['platform']} on {', '.join(runtime['hardware'])}",
        f"- Python {runtime['python']['min']} through {runtime['python']['max']} "
        f"(recommended {runtime['python']['recommended']})",
        f"- MLX >= {runtime['mlx']['min']} and < {runtime['mlx']['max_exclusive']}",
        f"- Canonical runtime entry point: `{runtime['entrypoint']}(...)`",
        "- Research and local evaluation workflows only",
        "",
        "## Model Support Matrix",
        "",
        "| Model family | Support status | Evidence depth | Notes |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for family in contract["families"]:
        lines.append(
            f"| {family['display_name']} | Allowlisted | {family['evidence_depth']} | "
            f"{family['workflow_story']} {family['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Secondary surfaces",
            "",
            "These surfaces exist, but they are not peer public runtime entry points:",
            "",
            "| Surface | Status | Why it is secondary | Preferred path |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )
    for surface in contract["secondary_surfaces"]:
        lines.append(
            f"| `{surface['path']}` | {surface['status']} | {surface['gate']} | `{surface['preferred']}` |"
        )

    lines.extend(
        [
            "",
            "## Release evidence contract",
            "",
            "A release claim is only addressable when the workflow publishes or references:",
            "",
        ]
    )
    for artifact in runtime["required_release_artifacts"]:
        lines.append(f"- `{artifact}`")

    lines.extend(
        [
            "",
            "## Benchmark publication rule",
            "",
            runtime["benchmark_publication_rule"],
            "",
            "Required provenance fields:",
            "",
        ]
    )
    for field in runtime["artifact_provenance_fields"]:
        lines.append(f"- `{field}`")
    return "\n".join(lines) + "\n"


def render_product_contract(contract: dict) -> str:
    runtime = contract["canonical_runtime"]
    lines = [
        GENERATED_HEADER,
        "# TurboQuant Product Contract",
        "",
        "This document defines the narrow supported surface TurboQuant can "
        "honestly claim today.",
        "",
        f"TurboQuant supports one canonical runtime path via `{runtime['entrypoint']}(...)`. "
        "A source archive documents that workflow, but it does not prove a current PASS "
        "without an addressable workflow artifact, release evidence bundle, or pinned "
        "manifest digest.",
        "",
        "## 1. Supported hardware and runtime",
        "",
        f"- Platform: `{runtime['platform']}`",
        f"- Hardware: {', '.join(runtime['hardware'])}",
        f"- Python: {runtime['python']['min']} to {runtime['python']['max']} "
        f"(recommended {runtime['python']['recommended']})",
        f"- MLX: >= {runtime['mlx']['min']} and < {runtime['mlx']['max_exclusive']}",
        "- Scope: local Apple-Silicon MLX validation, not production deployment",
        "",
        "## 2. Supported model families",
        "",
    ]
    for family in contract["families"]:
        lines.append(
            f"- **{family['display_name']}** — allowlisted; evidence depth is "
            f"**{family['evidence_depth']}**. {family['workflow_story']} {family['notes']} "
            f"Coverage: {', '.join(family['coverage'])}."
        )

    lines.extend(
        [
            "",
            "## 3. Canonical and secondary surfaces",
            "",
            f"- Canonical runtime path: `{runtime['entrypoint']}(...)`",
            "- Secondary surfaces remain available only for compatibility or eval use:",
        ]
    )
    for surface in contract["secondary_surfaces"]:
        lines.append(
            f"  - `{surface['path']}` ({surface['status']}) — {surface['gate']}"
        )

    lines.extend(
        [
            "",
            "## 4. Paper-facing presets and exact deviations",
            "",
            "Paper-facing presets are `paper_mse` and `paper_prod`/`paper_prod_qjl`. "
            "Legacy top-k presets remain compatibility paths, not the main algorithm story.",
            "",
        ]
    )
    for deviation in contract["deviations"]:
        lines.append(f"- **{deviation['title']}** — {deviation['description']}")

    lines.extend(
        [
            "",
            "## 5. Release evidence and benchmarks",
            "",
            runtime["source_archive_evidence_rule"],
            "",
            runtime["benchmark_publication_rule"],
            "",
            "Required release artifacts:",
            "",
        ]
    )
    for artifact in runtime["required_release_artifacts"]:
        lines.append(f"- `{artifact}`")
    return "\n".join(lines) + "\n"


def render_docs() -> dict[Path, str]:
    contract = load_contract()
    return {
        REPO_ROOT / "docs" / "support_matrix.md": render_support_matrix(contract),
        REPO_ROOT / "docs" / "supported-surface.md": render_supported_surface(contract),
        REPO_ROOT / "docs" / "product_contract.md": render_product_contract(contract),
    }


def write_artifact(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "contract.json"
    output_path.write_text(CONTRACT_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render contract-driven docs and artifacts")
    parser.add_argument("--check", action="store_true", help="Fail if rendered docs differ from checked-in docs")
    parser.add_argument("--artifact-dir", help="Optional artifact directory to receive contract.json")
    args = parser.parse_args()

    docs = render_docs()
    mismatches: list[str] = []
    for path, content in docs.items():
        if args.check:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            if existing != content:
                mismatches.append(str(path.relative_to(REPO_ROOT)))
        else:
            path.write_text(content, encoding="utf-8")

    if args.artifact_dir:
        write_artifact(Path(args.artifact_dir))

    if mismatches:
        raise SystemExit(
            "Rendered support-contract docs are out of date: " + ", ".join(mismatches)
        )


if __name__ == "__main__":
    main()