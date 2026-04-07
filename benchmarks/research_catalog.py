from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactSpec:
    path: str
    role: str


@dataclass(frozen=True)
class BenchmarkSurface:
    slug: str
    title: str
    lane: str
    metric_family: str
    script: str
    doc_path: str
    family_scope: tuple[str, ...]
    stable_outputs: tuple[ArtifactSpec, ...]
    smoke_test: str | None
    contract_effect: str
    summary: str


@dataclass(frozen=True)
class FamilyResearchAdjunct:
    family: str
    research_only_evidence: tuple[str, ...]
    still_unproven: tuple[str, ...]


BENCHMARK_SURFACES: tuple[BenchmarkSurface, ...] = (
    BenchmarkSurface(
        slug="dense_vs_turboquant_certification",
        title="Dense-vs-TurboQuant certification sweep",
        lane="product-certification",
        metric_family="runtime_certification",
        script="benchmarks/runtime_cert/run_dense_vs_tq.py",
        doc_path="docs/runtime-certification.md",
        family_scope=("llama", "gemma"),
        stable_outputs=(
            ArtifactSpec(
                path="aggregate_runs.csv",
                role="paired dense-vs-TurboQuant benchmark rows",
            ),
            ArtifactSpec(
                path="certification_summary.json",
                role="aggregate memory and throughput deltas",
            ),
        ),
        smoke_test=None,
        contract_effect=(
            "Can contribute to release evidence only when carried by the "
            "addressable certification bundle alongside cert_manifest.json and "
            "the required release artifacts."
        ),
        summary=(
            "Paired runtime-cert benchmark sweep for the bounded Apple-MLX "
            "product lane."
        ),
    ),
    BenchmarkSurface(
        slug="inner_product_bias",
        title="Inner-product bias summary",
        lane="research-only",
        metric_family="inner_product_bias",
        script="benchmarks/runtime_cert/run_inner_product_bias_eval.py",
        doc_path="docs/benchmark_methodology.md",
        family_scope=("synthetic",),
        stable_outputs=(
            ArtifactSpec(
                path="inner_product_bias_summary.json",
                role="structured research summary",
            ),
            ArtifactSpec(
                path="inner_product_bias_metrics.csv",
                role="row-oriented metric table",
            ),
            ArtifactSpec(
                path="inner_product_bias_summary.md",
                role="human-readable summary",
            ),
        ),
        smoke_test="tests/integration_research/test_inner_product_bias_artifacts.py",
        contract_effect=(
            "Research-only synthetic score diagnostic; it does not prove "
            "unbiasedness and it does not alter the release gate."
        ),
        summary=(
            "Synthetic scalar-only versus two-stage score-estimation comparison "
            "for the paper-facing path."
        ),
    ),
    BenchmarkSurface(
        slug="bit_budget_sweep",
        title="Bit-budget sweep",
        lane="research-only",
        metric_family="bit_budget_sweep",
        script="benchmarks/runtime_cert/run_bit_budget_sweep.py",
        doc_path="docs/bit_budget_sweep.md",
        family_scope=("synthetic",),
        stable_outputs=(
            ArtifactSpec(
                path="bit_budget_sweep_summary.json",
                role="structured research summary",
            ),
            ArtifactSpec(
                path="bit_budget_sweep_metrics.csv",
                role="row-oriented operating-point table",
            ),
            ArtifactSpec(
                path="bit_budget_sweep_summary.md",
                role="human-readable summary",
            ),
        ),
        smoke_test="tests/integration_research/test_bit_budget_sweep_smoke.py",
        contract_effect=(
            "Research-only synthetic operating-point comparison; it does not "
            "supply real-model quality evidence or expand support."
        ),
        summary=(
            "Synthetic comparison surface for compression, distortion, bias, "
            "and cache-update tradeoffs across stable presets."
        ),
    ),
    BenchmarkSurface(
        slug="kv_paper_eval",
        title="Unified KV paper evaluation bundle",
        lane="research-only",
        metric_family="kv_paper_eval",
        script="benchmarks/runtime_cert/run_kv_paper_eval.py",
        doc_path="docs/kv_paper_eval.md",
        family_scope=("llama", "gemma"),
        stable_outputs=(
            ArtifactSpec(
                path="kv_paper_eval_summary.json",
                role="structured stage-level research summary",
            ),
            ArtifactSpec(
                path="kv_paper_eval_summary.md",
                role="human-readable rollup",
            ),
        ),
        smoke_test="tests/integration_research/test_kv_paper_eval_smoke.py",
        contract_effect=(
            "Research-only consolidation of fast-check and optional heavy-offline "
            "KV evidence; it complements certification artifacts but never "
            "replaces them."
        ),
        summary=(
            "Unified report that makes lightweight runtime checks and optional "
            "heavier real-model research stages explicit in one bundle."
        ),
    ),
    BenchmarkSurface(
        slug="vector_search",
        title="Vector-search research lane",
        lane="research-only",
        metric_family="vector_search",
        script="benchmarks/vector_search/run_vector_search_eval.py",
        doc_path="docs/vector_search.md",
        family_scope=("not-applicable",),
        stable_outputs=(
            ArtifactSpec(
                path="vector_search_summary.json",
                role="structured research summary",
            ),
            ArtifactSpec(
                path="vector_search_metrics.csv",
                role="row-oriented retrieval metric table",
            ),
            ArtifactSpec(
                path="vector_search_summary.md",
                role="human-readable summary",
            ),
        ),
        smoke_test="tests/integration_research/test_vector_search_mini_smoke.py",
        contract_effect=(
            "Research-only retrieval benchmark; it stays outside the supported "
            "Apple-MLX product contract."
        ),
        summary=(
            "Bundled mini-dataset and explicit public-corpus retrieval lane for "
            "paper-adjacent search experiments."
        ),
    ),
)


FAMILY_RESEARCH_ADJUNCTS: tuple[FamilyResearchAdjunct, ...] = (
    FamilyResearchAdjunct(
        family="llama",
        research_only_evidence=(
            "The unified KV paper bundle can retain heavy-offline Llama dense-vs-TurboQuant and paper_mse quality stages as research context.",
            "Inner-product bias and bit-budget sweep remain synthetic adjuncts rather than Llama release gates.",
            "Vector-search remains a retrieval research lane and does not widen Llama runtime support.",
        ),
        still_unproven=(
            "No universal throughput-win claim on the current uncompiled Apple-MLX path.",
            "No theorem-level unbiased-inner-product claim for paper_prod_qjl.",
            "No support claim beyond the bounded Apple-Silicon allowlist and addressable release artifacts.",
        ),
    ),
    FamilyResearchAdjunct(
        family="gemma",
        research_only_evidence=(
            "The unified KV paper bundle can retain Gemma runtime fast-check stages and an optional observational paper_mse tranche.",
            "Gemma paper_mse research results stay observational and do not create symmetry with the Llama release guardrail.",
            "Inner-product bias, bit-budget sweep, and vector-search remain family-agnostic research context only.",
        ),
        still_unproven=(
            "Gemma does not have the same release-gated paper_mse quality depth as Llama.",
            "No universal throughput-win claim on the current uncompiled Apple-MLX path.",
            "No theorem-level unbiased-inner-product claim for paper_prod_qjl.",
        ),
    ),
)


def benchmark_surfaces() -> tuple[BenchmarkSurface, ...]:
    return BENCHMARK_SURFACES


def family_research_adjunct_map() -> dict[str, FamilyResearchAdjunct]:
    return {item.family: item for item in FAMILY_RESEARCH_ADJUNCTS}
