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
| `bench_dense_vs_turboquant.py` | Head-to-head memory and encode latency across bit-width configs | `artifacts/benchmarks/` |
| `bench_memory_footprint.py` | Per-buffer memory breakdown across token counts and configs | `artifacts/benchmarks/memory_footprint.txt` |
| `bench_decode_streaming.py` | Streaming attention decode latency vs dense baseline | `artifacts/benchmarks/decode_streaming.txt` |
| `bench_polar_vs_scalar.py` | PolarQuantizer vs GroupScalarQuantizer: encode/decode latency and MSE | `artifacts/benchmarks/polar_vs_scalar.json` |

**Measured numbers (Apple Silicon M-series, April 2026):**

```text
K-Encode Benchmark:       0.30 ms / step   (shape [1, 32, 128, 128], 100 iterations)
Decode Step Benchmark:    0.04 ms / step   (append_keys, 1 new token, 100 iterations)
```

**PolarQuant vs GroupScalarQuantizer (3-bit, group=64) — Apple Silicon M-series, April 2026:**

Config: batch=1, n_heads=8, reps=50, warmup=10.  All shapes [B·H·T, d_head].

```text
d_head    T    scalar enc ms  scalar dec ms  scalar MSE  scalar b/d    polar enc ms  polar dec ms   polar MSE  polar b/d  MSE ratio
    64  256    0.518±0.208    0.229±0.058    0.064083      3.250       0.043±0.004   0.361±0.141   0.038403     3.875      0.60x
    64  512    0.326±0.079    0.201±0.089    0.063783      3.250       0.049±0.007   0.309±0.089   0.038263     3.875      0.60x
    64 1024    0.376±0.095    0.196±0.071    0.063502      3.250       0.047±0.013   0.338±0.080   0.038252     3.875      0.60x
   128  256    0.334±0.084    0.193±0.060    0.063783      3.250       0.042±0.002   0.311±0.097   0.038263     3.875      0.60x
   128  512    0.376±0.092    0.199±0.083    0.063502      3.250       0.044±0.003   0.337±0.070   0.038252     3.875      0.60x
   128 1024    0.468±0.103    0.246±0.104    0.063460      3.250       0.042±0.001   0.493±0.081   0.038175     3.875      0.60x
   256  256    0.379±0.077    0.197±0.086    0.063502      3.250       0.051±0.020   0.331±0.071   0.038252     3.875      0.60x
   256  512    0.467±0.063    0.250±0.106    0.063460      3.250       0.044±0.006   0.520±0.118   0.038175     3.875      0.60x
   256 1024    0.912±0.094    0.339±0.085    0.063266      3.250       0.051±0.022   1.044±0.096   0.038119     3.875      0.60x
```

Key observations:
- PolarQuant **encode** is 7–18× faster (arctan2 + argmin vs bit-packing).
- PolarQuant **MSE** is uniformly 40% lower (0.60× ratio) at slightly higher bit-rate.
- PolarQuant **decode** is comparable or slightly slower at large T due to interleaved reconstruction.
- Full JSON results in `artifacts/benchmarks/polar_vs_scalar.json`.

**Dense vs TurboQuant — memory compression and encode latency:**

```text
config                          tokens  dense_MB    tq_MB   ratio   ms_dense    ms_tq
k_bits=4  k_group_size=64          256      0.52     0.07     7.5x      0.505    0.315
k_bits=4  k_group_size=64          512      1.05     0.14     7.5x      0.459    0.318
k_bits=4  k_group_size=64         1024      2.10     0.28     7.5x      0.499    0.401
k_bits=3  k_group_size=64          256      0.52     0.06     8.5x      0.475    0.315
k_bits=3  k_group_size=64          512      1.05     0.12     8.5x      0.549    0.347
k_bits=3  k_group_size=64         1024      2.10     0.25     8.5x      0.423    0.343
k_bits=2  k_group_size=64          256      0.52     0.04    14.2x      0.506    0.298
k_bits=2  k_group_size=64          512      1.05     0.07    14.2x      0.467    0.394
k_bits=2  k_group_size=64         1024      2.10     0.15    14.2x      0.464    0.331
k_bits=3  k_group_size=32          256      0.52     0.07     8.0x      0.510    0.357
k_bits=3  k_group_size=32          512      1.05     0.13     8.0x      0.445    0.387
k_bits=3  k_group_size=32         1024      2.10     0.26     8.0x      0.460    0.307
```

**Memory footprint — per-buffer breakdown (3-bit K, group=64, 1024 tokens):**

```text
type                      bits  group  tokens   total_MB   bytes/tok   vs_dense
dense (float16)             16     --    1024       2.10        2048       1.0x
TurboQuant k=4b g=64         4     64    1024       0.28         272       7.5x
TurboQuant k=3b g=64         3     64    1024       0.25         240       8.5x
TurboQuant k=2b g=64         2     64    1024       0.15         144      14.2x
TurboQuant k=4b g=32         4     32    1024       0.29         288       7.1x
TurboQuant k=3b g=32         3     32    1024       0.26         256       8.0x

  k_packed_main                      229.4 kB
  k_scales                            16.4 kB
  v_dense                           1048.6 kB
  TOTAL                             1294.3 kB
```

**Streaming decode latency vs dense baseline:**

```text
seq_len  block_tokens   ms_streaming   ms_baseline   speedup
------------------------------------------------------------
    128            64          0.498         0.199     0.40x
    128           128          0.434         0.224     0.52x
    256            64          0.476         0.201     0.42x
    256           128          0.391         0.210     0.54x
    512            64          0.548         0.225     0.41x
    512           128          0.514         0.274     0.53x
   1024            64          0.701         0.301     0.43x
   1024           128          0.667         0.248     0.37x
```

> Streaming attention overhead (~2x vs dense) is expected — each decode step
> dequantises key blocks on-the-fly in pure Python/MLX. Enabling `mx.compile`
> on inner loop functions or `TQ_USE_METAL=1` recovers most of this gap.

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
