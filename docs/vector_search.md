# Vector Search Research Lane

This document describes the research-only vector-search benchmark surface.

It exists to make the paper's broader retrieval framing measurable without
pretending that vector search is part of the supported product contract.

## Default Path

The repo ships a bundled mini dataset and a reproducible evaluation driver:

```bash
source .venv-cert311/bin/activate
python benchmarks/vector_search/run_vector_search_eval.py \
    --output-dir artifacts/runtime-cert/manual_vector_search
```

The default path uses:

- a bundled mini public dataset under `benchmarks/vector_search/data/`
- deterministic hashed bag-of-words embeddings
- a dense baseline plus compressed comparisons for:
  - `paper_mse`
  - `paper_prod_qjl`
  - `polarquant_exp`

## What It Reports

The evaluation emits:

- `vector_search_summary.json`
- `vector_search_metrics.csv`
- `vector_search_summary.md`

Metrics include:

- recall@1
- recall@3
- index memory footprint
- compression ratio
- indexing time
- query-time behavior

## Opt-In Larger Public Corpus

The driver also supports a larger public retrieval corpus through an explicit
download flag. The repo does not ship this third-party dataset.

```bash
source .venv-cert311/bin/activate
python benchmarks/vector_search/run_vector_search_eval.py \
    --dataset scifact \
    --download-public-corpus \
    --public-corpus-max-docs 512 \
    --public-corpus-max-queries 64 \
    --output-dir artifacts/runtime-cert/manual_vector_search_scifact
```

Notes:

- `--download-public-corpus` is required the first time so the repo does not
  silently fetch third-party content.
- `--public-corpus-max-docs` and `--public-corpus-max-queries` give a
  deterministic smaller slice when you want a quicker research run.
- Review the upstream dataset terms before using the downloaded data.

## What It Does Not Claim

- It does not make vector search part of the supported Apple-MLX product lane.
- It does not replace larger public retrieval benchmarks.
- It does not prove that a small bundled dataset generalizes to broader search
  workloads.
- It does not weaken the release gate.

## Interpretation

- Use the dense baseline as the uncompressed retrieval reference on the bundled
  dataset.
- Use the compressed rows to compare how recall, memory, and latency move under
  the current preset surfaces.
- Treat the entire lane as research-only evidence.
- If you need the results to show up in a dated history snapshot, pass the
  generated `vector_search_summary.json` to `scripts/render_benchmark_snapshot.py`
  with `--vector-search-summary`.
