#!/usr/bin/env python3
# flake8: noqa: E402
"""Run a unified research-only KV paper evaluation bundle.

This command consolidates the repo's existing KV-cache evidence surfaces into a
single report bundle while keeping the supported product contract unchanged.

Two tiers are reported explicitly:

- fast-check: tiny-model or lightweight runtime path checks that are intended to
  stay quick and reproducible.
- heavy-offline: real-model dense-vs-TurboQuant and quality-eval lanes that are
  slower, require explicit model ids, and should not be confused with the
  product release gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.runtime_cert.research_report_schema import (
    build_artifact_paths,
    build_research_report,
    build_run_id,
)
from benchmarks.runtime_cert.utils import collect_environment_metadata, ensure_artifact_dir, write_json

SUMMARY_JSON = "kv_paper_eval_summary.json"
SUMMARY_MD = "kv_paper_eval_summary.md"
FAST_CHECK_STAGES = (
    (
        "llama_runtime_smoke",
        "Llama runtime smoke",
        "tests/integration_mlx/test_llama_runtime_smoke.py",
        "junit_llama_runtime_smoke.xml",
    ),
    (
        "gemma_runtime_smoke",
        "Gemma runtime smoke",
        "tests/integration_mlx/test_gemma_runtime_smoke.py",
        "junit_gemma_runtime_smoke.xml",
    ),
    (
        "long_context_stability",
        "Long-context stability",
        "tests/integration_mlx/test_long_context_stability.py",
        "junit_long_context_stability.xml",
    ),
    (
        "dense_vs_paper_mse_275bpc",
        "Dense vs paper_mse 2.75 bpc",
        "tests/integration_mlx/test_dense_vs_paper_mse_275bpc.py",
        "junit_dense_vs_paper_mse_275bpc.xml",
    ),
    (
        "dense_vs_paper_mse_375bpc",
        "Dense vs paper_mse 3.75 bpc",
        "tests/integration_mlx/test_dense_vs_paper_mse_375bpc.py",
        "junit_dense_vs_paper_mse_375bpc.xml",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a unified research-only KV paper evaluation bundle."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive the KV evaluation bundle.",
    )
    parser.add_argument(
        "--include-heavy-offline",
        action="store_true",
        help="Run heavy offline real-model stages when model ids are configured.",
    )
    parser.add_argument(
        "--llama-model",
        default=os.environ.get("TQ_TEST_LLAMA_MODEL", ""),
        help="Optional real-model id for Llama heavy-offline stages.",
    )
    parser.add_argument(
        "--gemma-model",
        default=os.environ.get("TQ_TEST_GEMMA_MODEL", ""),
        help="Optional real-model id for Gemma heavy-offline stages.",
    )
    parser.add_argument(
        "--prompt-classes",
        nargs="+",
        default=["short", "medium", "long"],
        help="Prompt classes to use for heavy dense-vs-TurboQuant sweeps.",
    )
    parser.add_argument(
        "--quality-prompt-classes",
        nargs="+",
        default=["short", "medium"],
        help="Prompt classes to use for heavy quality evaluation.",
    )
    parser.add_argument(
        "--include-gemma-quality-research",
        action="store_true",
        help="Run a research-only observational paper_mse quality tranche for Gemma in the heavy-offline bundle.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _stage_stub(
    *,
    stage_id: str,
    name: str,
    tier: str,
    status: str,
    notes: list[str],
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "name": name,
        "tier": tier,
        "status": status,
        "notes": notes,
        "artifacts": artifacts or [],
    }


def _run_command(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _tail(text: str, limit: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-limit:]) if lines else ""


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_ROOT))
    except ValueError:
        return str(resolved)


def _parse_junit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "failed",
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "notes": ["JUnit XML was not produced."],
        }

    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return {
            "status": "failed",
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "notes": ["JUnit XML did not contain a testsuite element."],
        }

    tests = int(suite.get("tests", "0"))
    failures = int(suite.get("failures", "0"))
    errors = int(suite.get("errors", "0"))
    skipped = int(suite.get("skipped", "0"))

    if tests == 0 or (skipped > 0 and skipped == tests):
        status = "skipped"
    elif failures > 0 or errors > 0:
        status = "failed"
    else:
        status = "passed"

    return {
        "status": status,
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "notes": [],
    }


def _run_pytest_stage(
    *,
    stage_id: str,
    name: str,
    test_path: str,
    junit_name: str,
    output_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    junit_path = output_dir / junit_name
    command = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-q",
        "--tb=short",
        f"--junitxml={junit_path}",
    ]
    completed = _run_command(command, env=env)
    summary = _parse_junit(junit_path)
    stage = {
        "stage_id": stage_id,
        "name": name,
        "tier": "fast-check",
        "status": summary["status"],
        "command": " ".join(command),
        "returncode": completed.returncode,
        "artifacts": [_display_path(junit_path)] if junit_path.exists() else [],
        "notes": summary["notes"],
        "pytest": {
            "tests": summary["tests"],
            "failures": summary["failures"],
            "errors": summary["errors"],
            "skipped": summary["skipped"],
        },
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }
    if completed.returncode != 0 and stage["status"] == "passed":
        stage["status"] = "failed"
    return stage


def _run_dense_vs_tq_stage(
    *,
    family: str,
    model_id: str,
    prompt_classes: list[str],
    max_new_tokens: int,
    seed: int,
    output_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    family_dir = ensure_artifact_dir(output_dir / f"{family}_dense_vs_tq")
    stage_commands: list[str] = []
    stage_notes: list[str] = []
    status = "passed"

    for prompt_class in prompt_classes:
        prompt_file = f"benchmarks/runtime_cert/prompts/{prompt_class}.jsonl"
        command = [
            sys.executable,
            "benchmarks/runtime_cert/run_dense_vs_tq.py",
            "--model",
            model_id,
            "--prompt-file",
            prompt_file,
            "--prompt-class",
            prompt_class,
            "--output-dir",
            str(family_dir),
            "--max-new-tokens",
            str(max_new_tokens),
            "--seed",
            str(seed),
            "--mode",
            "both",
        ]
        completed = _run_command(command, env=env)
        stage_commands.append(" ".join(command))
        if completed.returncode != 0:
            status = "failed"
            stage_notes.append(
                f"dense-vs-TurboQuant run for prompt class '{prompt_class}' failed"
            )

    collect_command = [
        sys.executable,
        "benchmarks/runtime_cert/collect_metrics.py",
        "--input-dir",
        str(family_dir),
        "--output-dir",
        str(family_dir),
    ]
    collect_run = _run_command(collect_command, env=env)
    stage_commands.append(" ".join(collect_command))
    if collect_run.returncode != 0:
        status = "failed"
        stage_notes.append("aggregate metric collection failed")

    summary_path = family_dir / "certification_summary.json"
    summary_payload: dict[str, Any] | None = None
    if summary_path.is_file():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    stage: dict[str, Any] = {
        "stage_id": f"{family}_dense_vs_tq",
        "name": f"Dense-vs-TurboQuant sweep ({family})",
        "tier": "heavy-offline",
        "status": status,
        "commands": stage_commands,
        "artifacts": [
            _display_path(family_dir / "aggregate_runs.csv"),
            _display_path(summary_path),
        ],
        "notes": stage_notes,
        "model": model_id,
        "prompt_classes": prompt_classes,
    }
    if summary_payload is not None:
        memory = summary_payload.get("memory_deltas", [])
        speed = summary_payload.get("speed_deltas", [])
        if memory:
            stage["memory_reduction_pct_range"] = {
                "min": min(float(item["reduction_pct"]) for item in memory),
                "max": max(float(item["reduction_pct"]) for item in memory),
            }
        if speed:
            stage["throughput_delta_pct_range"] = {
                "min": min(float(item["delta_pct"]) for item in speed),
                "max": max(float(item["delta_pct"]) for item in speed),
            }
    return stage


def _run_quality_eval_stage(
    *,
    stage_id: str,
    name: str,
    family: str,
    model_id: str,
    prompt_classes: list[str],
    output_dir: Path,
    env: dict[str, str],
    artifact_label: str = "",
    max_delta_ppl: float = 20.0,
    max_mean_kl: float = 5.0,
    research_only: bool = False,
) -> dict[str, Any]:
    family_dir = ensure_artifact_dir(output_dir / f"{family}_quality_eval")
    stage_commands: list[str] = []
    stage_notes: list[str] = []
    status = "passed"
    summaries: list[dict[str, Any]] = []
    artifacts: list[str] = []
    observation_statuses: list[str] = []

    for prompt_class in prompt_classes:
        prompt_file = f"benchmarks/runtime_cert/prompts/{prompt_class}.jsonl"
        command = [
            sys.executable,
            "benchmarks/runtime_cert/run_quality_eval.py",
            "--model",
            model_id,
            "--prompt-file",
            prompt_file,
            "--prompt-class",
            prompt_class,
            "--output-dir",
            str(family_dir),
            "--preset",
            "paper_mse",
            "--model-family",
            family,
            "--min-prompt-tokens",
            "32",
            "--max-delta-ppl",
            str(max_delta_ppl),
            "--max-mean-kl",
            str(max_mean_kl),
            "--seed",
            "42",
        ]
        if artifact_label:
            command.extend(["--artifact-label", artifact_label])
        completed = _run_command(command, env=env)
        stage_commands.append(" ".join(command))
        summary_name = (
            f"quality_eval_{artifact_label}_{prompt_class}_summary.json"
            if artifact_label
            else f"quality_eval_{prompt_class}_summary.json"
        )
        summary_path = family_dir / summary_name
        if summary_path.is_file():
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries.append(summary_payload)
            artifacts.append(_display_path(summary_path))
            observation_statuses.append(str(summary_payload.get("status", "unknown")))
        if completed.returncode != 0 and not research_only:
            status = "failed"
            stage_notes.append(f"quality eval for prompt class '{prompt_class}' failed")

    if research_only:
        if len(summaries) == len(prompt_classes):
            status = "captured"
        else:
            status = "failed"
        observed = ", ".join(
            f"{prompt_class}:{observation_status}"
            for prompt_class, observation_status in zip(prompt_classes, observation_statuses)
        )
        stage_notes.append(
            "Research-only observational tranche; this does not promote Gemma to a symmetric release guardrail."
        )
        if observed:
            stage_notes.append(f"Observed paper_mse gate outcomes: {observed}.")

    stage: dict[str, Any] = {
        "stage_id": stage_id,
        "name": name,
        "tier": "heavy-offline",
        "status": status,
        "commands": stage_commands,
        "artifacts": artifacts,
        "notes": stage_notes,
        "model": model_id,
        "prompt_classes": prompt_classes,
        "evaluation_mode": (
            "research-only-observational" if research_only else "guardrail"
        ),
        "thresholds": {
            "max_delta_ppl": max_delta_ppl,
            "max_mean_kl": max_mean_kl,
        },
    }
    if summaries:
        stage["quality_summary"] = {
            "mean_delta_ppl": sum(float(item["mean_delta_ppl"]) for item in summaries)
            / len(summaries),
            "mean_kl": sum(float(item["mean_kl"]) for item in summaries)
            / len(summaries),
        }
        stage["observation_statuses"] = observation_statuses
    return stage


def _render_markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# KV Paper Evaluation Summary",
        "",
        "This is a research-only consolidation of the repo's KV-cache evidence surfaces.",
        "It does not replace the Apple runtime certification gate and it does not widen the supported product contract.",
        "",
        "## Report Identity",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- preset group: `{payload['preset']}`",
        f"- family scope: `{payload['family']}`",
        "",
        "## Tier Definitions",
        "",
        "- `fast-check` — tiny-model or lightweight runtime path checks intended to stay quick and reproducible.",
        "- `heavy-offline` — real-model benchmark or quality-eval stages that require explicit model ids and can take substantially longer.",
        "",
        "## Stage Overview",
        "",
        "| Stage | Tier | Status | Key outputs | Notes |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for stage in payload["stages"]:
        artifacts = ", ".join(f"`{Path(path).name}`" for path in stage.get("artifacts", []))
        note = "; ".join(stage.get("notes", [])) or "-"
        lines.append(
            f"| {stage['name']} | `{stage['tier']}` | `{stage['status']}` | {artifacts or '-'} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Honest Takeaways",
            "",
            f"- Fast-check stages executed: `{payload['tier_counts']['fast-check']['executed']}`; heavy-offline stages executed: `{payload['tier_counts']['heavy-offline']['executed']}`.",
            "- Heavy-offline stages are reported explicitly as `not_requested` or `not_configured` when they are not run, so the bundle does not pretend to have evidence it never collected.",
            "- Fast-check passes show that the runtime path and tiny-model paper-facing comparisons still work end to end; they are not real-model quality or benchmark claims.",
            "- Heavy-offline results, when present, remain research evidence and should still be read alongside the narrower supported-family contract.",
            "",
            "## Companion Artifacts",
            "",
            f"- `{SUMMARY_JSON}` — structured stage summary",
            f"- `{SUMMARY_MD}` — human-readable summary",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    output_dir = ensure_artifact_dir(args.output_dir)
    environment = collect_environment_metadata(
        model="kv-paper-eval",
        mode="kv_paper_eval",
    )
    configured_families = [
        family
        for family, model_id in (("llama", args.llama_model), ("gemma", args.gemma_model))
        if model_id
    ]
    family_scope = (
        configured_families[0]
        if len(configured_families) == 1
        else ("mixed" if configured_families else "not-configured")
    )
    run_id = build_run_id(
        timestamp=str(environment["timestamp"]),
        label=f"{family_scope}_{'heavy' if args.include_heavy_offline else 'fast'}",
        mode="kv_paper_eval",
    )
    base_env = os.environ.copy()
    fast_env = base_env.copy()
    fast_env.pop("TQ_TEST_LLAMA_MODEL", None)
    fast_env.pop("TQ_TEST_GEMMA_MODEL", None)
    fast_env.pop("TQ_RUN_EXPLORATORY_REAL_MODEL_QUALITY", None)

    heavy_env = base_env.copy()
    if args.llama_model:
        heavy_env["TQ_TEST_LLAMA_MODEL"] = args.llama_model
    if args.gemma_model:
        heavy_env["TQ_TEST_GEMMA_MODEL"] = args.gemma_model

    stages: list[dict[str, Any]] = []
    fast_dir = ensure_artifact_dir(output_dir / "fast_checks")
    for stage_id, name, test_path, junit_name in FAST_CHECK_STAGES:
        stages.append(
            _run_pytest_stage(
                stage_id=stage_id,
                name=name,
                test_path=test_path,
                junit_name=junit_name,
                output_dir=fast_dir,
                env=fast_env,
            )
        )

    heavy_dir = ensure_artifact_dir(output_dir / "heavy_offline")
    if not args.include_heavy_offline:
        stages.append(
            _stage_stub(
                stage_id="heavy_offline_bundle",
                name="Heavy offline bundle",
                tier="heavy-offline",
                status="not_requested",
                notes=["Run again with --include-heavy-offline to collect real-model sweeps and quality evaluation."],
            )
        )
    else:
        if args.llama_model:
            stages.append(
                _run_dense_vs_tq_stage(
                    family="llama",
                    model_id=args.llama_model,
                    prompt_classes=list(args.prompt_classes),
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    output_dir=heavy_dir,
                    env=heavy_env,
                )
            )
            stages.append(
                _run_quality_eval_stage(
                    stage_id="llama_quality_eval",
                    name="paper_mse quality evaluation (llama)",
                    family="llama",
                    model_id=args.llama_model,
                    prompt_classes=list(args.quality_prompt_classes),
                    output_dir=heavy_dir,
                    env=heavy_env,
                )
            )
        else:
            stages.append(
                _stage_stub(
                    stage_id="llama_heavy_bundle",
                    name="Llama heavy offline bundle",
                    tier="heavy-offline",
                    status="not_configured",
                    notes=["No --llama-model provided, so real-model Llama stages were not run."],
                )
            )

        if args.gemma_model:
            stages.append(
                _run_dense_vs_tq_stage(
                    family="gemma",
                    model_id=args.gemma_model,
                    prompt_classes=list(args.prompt_classes),
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    output_dir=heavy_dir,
                    env=heavy_env,
                )
            )
            if args.include_gemma_quality_research:
                stages.append(
                    _run_quality_eval_stage(
                        stage_id="gemma_quality_eval_research",
                        name="paper_mse quality evaluation (gemma research-only)",
                        family="gemma",
                        model_id=args.gemma_model,
                        prompt_classes=list(args.quality_prompt_classes),
                        output_dir=heavy_dir,
                        env=heavy_env,
                        artifact_label="gemma_research",
                        research_only=True,
                    )
                )
            else:
                stages.append(
                    _stage_stub(
                        stage_id="gemma_quality_eval_research",
                        name="paper_mse quality evaluation (gemma research-only)",
                        tier="heavy-offline",
                        status="not_requested",
                        notes=[
                            "Gemma paper_mse quality remains a research-only observational tranche. "
                            "Run again with --include-gemma-quality-research to collect it explicitly."
                        ],
                    )
                )
        else:
            stages.append(
                _stage_stub(
                    stage_id="gemma_heavy_bundle",
                    name="Gemma heavy offline bundle",
                    tier="heavy-offline",
                    status="not_configured",
                    notes=["No --gemma-model provided, so real-model Gemma stages were not run."],
                )
            )

    tier_counts = {
        "fast-check": {
            "executed": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "captured": 0,
        },
        "heavy-offline": {
            "executed": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "captured": 0,
        },
    }
    overall_status = "passed"
    for stage in stages:
        tier = stage["tier"]
        status = stage["status"]
        if status in {"passed", "failed", "skipped", "captured"}:
            tier_counts[tier]["executed"] += 1
            tier_counts[tier][status] += 1
        if status == "failed":
            overall_status = "failed"

    artifact_paths = build_artifact_paths(
        summary_json=SUMMARY_JSON,
        summary_markdown=SUMMARY_MD,
    )

    payload = build_research_report(
        schema_version="1",
        metric_family="kv_paper_eval",
        run_id=run_id,
        environment=environment,
        preset="mixed",
        family=family_scope,
        scope="research-only",
        mode="kv_paper_eval",
        status=overall_status,
        metrics={
            "fast_check_executed": tier_counts["fast-check"]["executed"],
            "heavy_offline_executed": tier_counts["heavy-offline"]["executed"],
            "heavy_offline_captured": tier_counts["heavy-offline"]["captured"],
        },
        artifact_paths=artifact_paths,
        notes=[
            "This bundle consolidates existing KV runtime, stability, and benchmark surfaces into one research-only report.",
            "Fast-check stages are intentionally lighter than heavy-offline real-model sweeps.",
            "Heavy-offline stages that are not run stay explicit in the report instead of being silently omitted.",
        ],
        support_scope="research-only",
        heavy_offline_requested=args.include_heavy_offline,
        llama_model=args.llama_model or None,
        gemma_model=args.gemma_model or None,
        tier_counts=tier_counts,
        stages=stages,
        companion_artifacts=[SUMMARY_JSON, SUMMARY_MD],
    )

    summary_path = output_dir / SUMMARY_JSON
    markdown_path = output_dir / SUMMARY_MD
    write_json(summary_path, payload)
    markdown_path.write_text(_render_markdown_summary(payload), encoding="utf-8")

    print(summary_path)
    print(markdown_path)
    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
