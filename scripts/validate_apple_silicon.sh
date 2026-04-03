#!/usr/bin/env bash
# scripts/validate_apple_silicon.sh — local Apple Silicon validation lane.
#
# This script runs the checks that are meaningful on Apple Silicon but safe to
# run without real model weights.  It does NOT constitute runtime certification
# (use scripts/certify_apple_runtime.sh for that).
#
# Lanes:
#   1. Strict preflight        — environment / dependency check
#   2. MLX unit tests          — fast, synthetic-tensor tests for TQ core path
#   3. Path-proof gate         — structural proof: TQ path is active, not dense
#   4. Cache + attention tests — roundtrip and equivalence (no model weights)
#   5. Model smoke tests       — OPTIONAL, requires TQ_TEST_LLAMA_MODEL /
#                                TQ_TEST_GEMMA_MODEL env vars
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools nox
python -m pip install -e '.[dev,apple]'

echo ""
echo "──── Lane 1: Strict preflight ────"
python3 scripts/preflight.py --strict

echo ""
echo "──── Lane 2: MLX unit tests ────"
python3 -m pytest tests/unit_mlx/ -q --tb=short

echo ""
echo "──── Lane 3: Path-proof gate (structural — no model weights) ────"
python3 -m pytest tests/integration_mlx/test_path_not_dense_fallback.py -v --tb=short

echo ""
echo "──── Lane 4: Cache roundtrip + attention equivalence ────"
python3 -m pytest \
    tests/integration_mlx/test_cache_upgrade_roundtrip.py \
    tests/integration_mlx/test_streaming_attention_equivalence.py \
    -v --tb=short

if [ -n "${TQ_TEST_LLAMA_MODEL:-}" ] || [ -n "${TQ_TEST_GEMMA_MODEL:-}" ]; then
    echo ""
    echo "──── Lane 5: Model smoke tests (optional) ────"
    SMOKE_ARGS=""
    if [ -n "${TQ_TEST_LLAMA_MODEL:-}" ]; then
        SMOKE_ARGS="$SMOKE_ARGS tests/integration_mlx/test_llama_runtime_smoke.py"
    fi
    if [ -n "${TQ_TEST_GEMMA_MODEL:-}" ]; then
        SMOKE_ARGS="$SMOKE_ARGS tests/integration_mlx/test_gemma_runtime_smoke.py"
    fi
    # shellcheck disable=SC2086
    python3 -m pytest $SMOKE_ARGS -v --tb=short
else
    echo ""
    echo "──── Lane 5: Model smoke tests ────"
    echo "  SKIPPED (set TQ_TEST_LLAMA_MODEL and/or TQ_TEST_GEMMA_MODEL to enable)"
fi
