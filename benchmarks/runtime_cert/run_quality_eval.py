"""
run_quality_eval.py — TurboQuant quality evaluation gate.

Usage::

    python benchmarks/runtime_cert/run_quality_eval.py \
        --model mlx-community/Llama-3.2-1B-Instruct-4bit \
        --prompt-file benchmarks/runtime_cert/prompts/wikitext.jsonl \
        --output-dir artifacts/runtime-cert/20260401_120000 \
        --max-delta-ppl 0.5 \
        --max-mean-kl 0.1

Each line of the JSONL prompt file must have a ``"prompt"`` or ``"text"`` key.
The script exits 0 (PASS) when *all* quality gates are satisfied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_prompts(prompt_file: Path) -> list[str]:
    """Read a JSONL file and return a list of prompt strings."""
    prompts = []
    with prompt_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("prompt") or rec.get("text") or ""
            if text:
                prompts.append(text)
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def _kl_div(log_p: "mx.array", log_q: "mx.array") -> float:
    """Mean per-token KL(P_dense || P_tq) from log-softmax arrays [T, V]."""
    import mlx.core as mx
    # KL(P || Q) = sum_v P_v (log P_v - log Q_v)
    p = mx.exp(log_p)
    kl_per_token = mx.sum(p * (log_p - log_q), axis=-1)  # [T]
    return float(mx.mean(kl_per_token).item())


def _eval_one_prompt(
    model,
    input_ids: "mx.array",
    turboquant_config,
) -> dict:
    """Run dense and TurboQuant forward passes; return metrics for one prompt."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from turboquant.integrations.mlx.upgrade import upgrade_cache_list

    targets = input_ids[0, 1:]   # [T-1]
    feed = input_ids[:, :-1]      # [1, T-1]
    T = int(targets.shape[0])

    if T == 0:
        return {}

    # ── Dense forward ──────────────────────────────────────────────────────
    dense_cache = make_prompt_cache(model)
    dense_logits = model(feed, cache=dense_cache)[0]      # [T-1, V]
    mx.eval(dense_logits)
    dense_log_p = dense_logits - mx.logsumexp(dense_logits, axis=-1, keepdims=True)
    dense_nll = -float(mx.sum(dense_log_p[mx.arange(T), targets]).item())
    dense_ppl = math.exp(dense_nll / T)

    # ── TurboQuant forward ─────────────────────────────────────────────────
    tq_cache = make_prompt_cache(model)
    upgrade_cache_list(tq_cache, k_start=0, config=turboquant_config)
    tq_logits = model(feed, cache=tq_cache)[0]            # [T-1, V]
    mx.eval(tq_logits)
    tq_log_p = tq_logits - mx.logsumexp(tq_logits, axis=-1, keepdims=True)
    tq_nll = -float(mx.sum(tq_log_p[mx.arange(T), targets]).item())
    tq_ppl = math.exp(tq_nll / T)

    kl = _kl_div(dense_log_p, tq_log_p)

    return {
        "n_tokens": T,
        "dense_ppl": round(dense_ppl, 4),
        "tq_ppl": round(tq_ppl, 4),
        "delta_ppl": round(tq_ppl - dense_ppl, 4),
        "mean_kl": round(kl, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant quality evaluation gate")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--prompt-file", required=True, help="Path to a .jsonl prompt file"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for artifacts"
    )
    parser.add_argument(
        "--max-delta-ppl", type=float, default=0.5,
        help="Maximum allowed mean TurboQuant PPL increase (default: 0.5)"
    )
    parser.add_argument(
        "--max-mean-kl", type=float, default=0.1,
        help="Maximum allowed mean per-token KL divergence (default: 0.1)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    from mlx_lm import load as mlx_load  # type: ignore
    model, tokenizer = mlx_load(args.model)

    from turboquant.config import TurboQuantConfig
    tq_config = TurboQuantConfig.from_preset("paper_prod")

    prompts = _load_prompts(Path(args.prompt_file))
    print(f"Evaluating {len(prompts)} prompt(s) …")

    results = []
    for i, text in enumerate(prompts):
        tokens = tokenizer.encode(text)
        import mlx.core as mx
        input_ids = mx.array(tokens)[None]   # [1, T]
        metrics = _eval_one_prompt(model, input_ids, tq_config)
        if metrics:
            results.append(metrics)
            print(
                f"  [{i+1}/{len(prompts)}] Δppl={metrics['delta_ppl']:+.3f}  "
                f"KL={metrics['mean_kl']:.5f}  "
                f"(dense={metrics['dense_ppl']:.2f}, tq={metrics['tq_ppl']:.2f})"
            )

    if not results:
        print("ERROR: no valid prompts produced results.", file=sys.stderr)
        sys.exit(2)

    mean_delta_ppl = sum(r["delta_ppl"] for r in results) / len(results)
    mean_kl = sum(r["mean_kl"] for r in results) / len(results)

    passed = (mean_delta_ppl <= args.max_delta_ppl) and (mean_kl <= args.max_mean_kl)
    status = "PASS" if passed else "FAIL"

    summary = {
        "status": status,
        "model": args.model,
        "n_prompts": len(results),
        "mean_delta_ppl": round(mean_delta_ppl, 4),
        "mean_kl": round(mean_kl, 6),
        "thresholds": {
            "max_delta_ppl": args.max_delta_ppl,
            "max_mean_kl": args.max_mean_kl,
        },
        "per_prompt": results,
    }

    artifact_path = output_dir / "quality_eval_summary.json"
    artifact_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*60}")
    print(f"  Status        : {status}")
    print(f"  Mean Δppl     : {mean_delta_ppl:+.4f}  (threshold ≤ {args.max_delta_ppl})")
    print(f"  Mean KL       : {mean_kl:.6f}  (threshold ≤ {args.max_mean_kl})")
    print(f"  Artifact      : {artifact_path}")
    print(f"{'='*60}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

