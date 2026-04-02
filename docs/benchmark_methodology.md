# Benchmark Methodology

## Execution rules

- Use `mx.eval()` to synchronize compute before timing.
- Use `time.perf_counter()` for wall-clock precision.
- Seed all runs with `mx.random.seed(42)` for reproducibility.

## Synthetic micro-benchmarks

Located in `benchmarks/exploratory/`. Run individually or via `scripts/run_benchmarks.sh`.

| Script | What it measures | Output |
|---|---|---|
| `bench_k_encode.py` | `encode_k_block` latency (100-step average) | `artifacts/benchmarks/k_encode.txt` |
| `bench_decode_step.py` | `cache.append_keys` latency per decode step | `artifacts/benchmarks/decode.txt` |
| `bench_memory.py` | Theoretical memory footprint summary | `artifacts/benchmarks/memory.txt` |

**Measured numbers (Apple Silicon, commit `6afc966`):**

```text
K-Encode Benchmark:    0.10 ms / step   (shape [1, 32, 128, 128], 100 iterations)
Decode Step Benchmark: 0.03 ms / step   (append_keys, 1 new token, 100 iterations)
```

## Paired generative benchmarks

Located in `benchmarks/runtime_cert/`. Runs paired dense + TurboQuant generation and writes structured JSON artifacts.

```bash
python benchmarks/runtime_cert/run_dense_vs_tq.py \
    --model <hf-model-id> \
    --prompt-file benchmarks/runtime_cert/prompts/short.jsonl \
    --prompt-class short \
    --output-dir artifacts/run_full \
    --max-new-tokens 64 --seed 42 --mode both
```

Prompt classes: `short` (5 prompts), `medium` (5 prompts), `long` (5 prompts).

**Measured numbers (Apple Silicon, Llama/Gemma models, 64 tokens):**

```text
[dense]       avg 0.52 s  |  147–163 tok/s
[turboquant]  avg 6.80 s  |    9–10 tok/s   ← Python streaming path (uncompiled)
```

> TurboQuant decode speed reflects the uncompiled Python streaming attention path.
> Enable `TQ_USE_METAL=1` or wrap inner functions with `mx.compile` for production throughput.
