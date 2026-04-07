<!-- Generated from benchmarks/research_catalog.py and turboquant/contract.json by scripts/render_research_docs.py. Do not edit by hand. -->
# Benchmark Index

This generated index maps the repo's retained benchmark and report surfaces to their lane, stable outputs, and proof discipline.
It keeps product certification evidence separate from research-only evidence and does not widen the supported product contract.
Working trees may retain generated `artifacts/runtime-cert/` bundles for archaeology, but built wheels and source distributions do not ship those directories. No source or built snapshot proves a current PASS unless it is accompanied by an addressable workflow artifact, release evidence bundle, or pinned manifest digest.

## Lane Boundary

- Product-certification surfaces can contribute to release truth only when they travel inside an addressable certification bundle with the required manifest and provenance fields.
- Research-only surfaces emit stable report bundles for comparison and archaeology, but they do not weaken the release gate or promote research into support truth.

## Product-Certification Surfaces

| Surface | Primary script | Family scope | Stable outputs | Release role | Primary doc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dense-vs-TurboQuant certification sweep | `benchmarks/runtime_cert/run_dense_vs_tq.py` | `llama`, `gemma` | `aggregate_runs.csv`<br>`certification_summary.json` | Can contribute to release evidence only when carried by the addressable certification bundle alongside cert_manifest.json and the required release artifacts. | [docs/runtime-certification.md](docs/runtime-certification.md) |

The certification workflow also carries runtime smokes, long-context stability, and family-scoped quality guardrails. Those stages are defined in [docs/runtime-certification.md](docs/runtime-certification.md) and remain the product lane's release gate.

## Research-Only Report Surfaces

| Surface | Metric family | Primary script | Stable outputs | Deterministic smoke | Scope note | Primary doc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Inner-product bias summary | `inner_product_bias` | `benchmarks/runtime_cert/run_inner_product_bias_eval.py` | `inner_product_bias_summary.json`<br>`inner_product_bias_metrics.csv`<br>`inner_product_bias_summary.md` | `tests/integration_research/test_inner_product_bias_artifacts.py` | Research-only synthetic score diagnostic; it does not prove unbiasedness and it does not alter the release gate. | [docs/benchmark_methodology.md](docs/benchmark_methodology.md) |
| Bit-budget sweep | `bit_budget_sweep` | `benchmarks/runtime_cert/run_bit_budget_sweep.py` | `bit_budget_sweep_summary.json`<br>`bit_budget_sweep_metrics.csv`<br>`bit_budget_sweep_summary.md` | `tests/integration_research/test_bit_budget_sweep_smoke.py` | Research-only synthetic operating-point comparison; it does not supply real-model quality evidence or expand support. | [docs/bit_budget_sweep.md](docs/bit_budget_sweep.md) |
| Unified KV paper evaluation bundle | `kv_paper_eval` | `benchmarks/runtime_cert/run_kv_paper_eval.py` | `kv_paper_eval_summary.json`<br>`kv_paper_eval_summary.md` | `tests/integration_research/test_kv_paper_eval_smoke.py` | Research-only consolidation of fast-check and optional heavy-offline KV evidence; it complements certification artifacts but never replaces them. | [docs/kv_paper_eval.md](docs/kv_paper_eval.md) |
| Vector-search research lane | `vector_search` | `benchmarks/vector_search/run_vector_search_eval.py` | `vector_search_summary.json`<br>`vector_search_metrics.csv`<br>`vector_search_summary.md` | `tests/integration_research/test_vector_search_mini_smoke.py` | Research-only retrieval benchmark; it stays outside the supported Apple-MLX product contract. | [docs/vector_search.md](docs/vector_search.md) |

## Common Research Report Envelope

The research-only report writers share one stable top-level envelope from `benchmarks/runtime_cert/research_report_schema.py` so JSON payloads stay comparable across commits.

| Field | Meaning |
| :--- | :--- |
| `schema_version` | stable research-report envelope version |
| `metric_family` | benchmark/report family identifier |
| `run_id` | timestamp-derived run label for cross-artifact alignment |
| `timestamp` | captured environment timestamp |
| `preset` | primary preset or preset group for the report |
| `family` | family scope or synthetic/not-applicable scope label |
| `scope` | lane label; research reports stay research-only |
| `mode` | script-level mode identifier |
| `status` | top-level run outcome |
| `metrics` | compact summary metrics for quick comparisons |
| `artifact_paths` | stable companion artifact filenames |
| `notes` | honest interpretation notes retained with the report |

## Notes

- `inner_product_bias` and `bit_budget_sweep` are synthetic by design and remain outside family-level runtime proof.
- `kv_paper_eval` explicitly distinguishes `fast-check` from `heavy-offline` stages so missing heavy evidence is never silently fabricated.
- `vector_search` remains a research lane even when its summary is carried alongside a dated benchmark snapshot.
