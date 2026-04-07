<!-- Generated from benchmarks/research_catalog.py and turboquant/contract.json by scripts/render_research_docs.py. Do not edit by hand. -->
# Family Evidence Matrix

This generated matrix separates release-gated family evidence from research-only adjunct evidence.
It preserves the current asymmetry between Llama and Gemma instead of flattening them into one support depth.
Working trees may retain generated `artifacts/runtime-cert/` bundles for archaeology, but built wheels and source distributions do not ship those directories. No source or built snapshot proves a current PASS unless it is accompanied by an addressable workflow artifact, release evidence bundle, or pinned manifest digest.

| Family | Product evidence depth | Release-gated evidence | Research-only adjunct evidence | Still unproven |
| :--- | :--- | :--- | :--- | :--- |
| Llama | stronger | real-model smoke<br>PolarQuant runtime smoke<br>PolarQuant quality guardrail<br>batch quality guardrail<br>long-context stability<br>dense-vs-TurboQuant benchmark sweeps | The unified KV paper bundle can retain heavy-offline Llama dense-vs-TurboQuant and paper_mse quality stages as research context.<br>Inner-product bias and bit-budget sweep remain synthetic adjuncts rather than Llama release gates.<br>Vector-search remains a retrieval research lane and does not widen Llama runtime support. | Source archives alone do not prove a current PASS for Llama; use an addressable evidence bundle or pinned manifest digest.<br>No universal throughput-win claim on the current uncompiled Apple-MLX path.<br>No theorem-level unbiased-inner-product claim for paper_prod_qjl.<br>No support claim beyond the bounded Apple-Silicon allowlist and addressable release artifacts. |
| Gemma | narrower | real-model smoke<br>PolarQuant runtime smoke<br>PolarQuant quality guardrail<br>dense-vs-TurboQuant benchmark sweeps | The unified KV paper bundle can retain Gemma runtime fast-check stages and an optional observational paper_mse tranche.<br>Gemma paper_mse research results stay observational and do not create symmetry with the Llama release guardrail.<br>Inner-product bias, bit-budget sweep, and vector-search remain family-agnostic research context only. | Gemma coverage remains narrower overall because the conservative paper_mse batch quality guardrail remains Llama-scoped, even though PolarQuant runtime and quality evidence now exist for Gemma; source archives alone do not prove a current PASS.<br>Gemma does not have the same release-gated paper_mse quality depth as Llama.<br>No universal throughput-win claim on the current uncompiled Apple-MLX path.<br>No theorem-level unbiased-inner-product claim for paper_prod_qjl. |

## Family-Agnostic Research Lanes

| Surface | Metric family | Family scope | Why it does not change family evidence depth |
| :--- | :--- | :--- | :--- |
| Inner-product bias summary | `inner_product_bias` | `synthetic` | Research-only synthetic score diagnostic; it does not prove unbiasedness and it does not alter the release gate. |
| Bit-budget sweep | `bit_budget_sweep` | `synthetic` | Research-only synthetic operating-point comparison; it does not supply real-model quality evidence or expand support. |
| Vector-search research lane | `vector_search` | `not-applicable` | Research-only retrieval benchmark; it stays outside the supported Apple-MLX product contract. |

## Current Asymmetry

- Llama retains the stronger release-gated evidence depth in the current contract.
- Gemma remains narrower because the conservative paper_mse quality guardrail is not symmetric with Llama.
- Synthetic and retrieval research lanes can inform future work, but they do not by themselves produce runtime-complete family proof.
