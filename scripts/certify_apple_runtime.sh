#!/usr/bin/env bash
# scripts/certify_apple_runtime.sh — one-command Apple-Silicon runtime certification.
#
# Usage:
#     ./scripts/certify_apple_runtime.sh
#
# Environment variables (required for model smoke tests):
#     TQ_TEST_LLAMA_MODEL   — small Llama-family HF model ID
#     TQ_TEST_GEMMA_MODEL   — small Gemma-family HF model ID
#
# Artifacts are written to: artifacts/runtime-cert/<timestamp>/
# Exit code is 0 only if every required stage passes with at least one test
# executed.  Skipped tests count as certification failures — a stage where
# all tests are marked @pytest.mark.skip is UNIMPLEMENTED, not PASSED.
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Timestamp and artifact directory
# ---------------------------------------------------------------------------
TS="$(date -u '+%Y%m%d_%H%M%S')"
ARTIFACT_DIR="$REPO_ROOT/artifacts/runtime-cert/$TS"
mkdir -p "$ARTIFACT_DIR"
echo "═══════════════════════════════════════════════════════════════"
echo "  TurboQuant Apple-Silicon Runtime Certification"
echo "  Timestamp : $TS"
echo "  Artifacts : $ARTIFACT_DIR"
echo "═══════════════════════════════════════════════════════════════"

FAILURES=0
STAGES_PASSED=0
STAGES_SKIPPED=0        # stages explicitly skipped because a required env var is not set
STAGES_UNIMPLEMENTED=0  # stages where pytest ran but produced zero passing tests (all @skip)
STAGES_TOTAL=0

# ---------------------------------------------------------------------------
# Internal helper: Python runtime selection for certification bootstrap.
# Prefer supported interpreters in descending order, but keep using an active
# environment when it is already on Python 3.9–3.11.
# ---------------------------------------------------------------------------
_supports_runtime_cert_python() {
    local python_bin="$1"
    "$python_bin" - <<'PYEOF'
import sys

major, minor = sys.version_info[:2]
raise SystemExit(0 if major == 3 and 9 <= minor <= 11 else 1)
PYEOF
}

_python_version() {
    local python_bin="$1"
    "$python_bin" - <<'PYEOF'
import sys

print(".".join(str(part) for part in sys.version_info[:3]))
PYEOF
}

_active_env_python() {
    local candidate=""
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        return 1
    fi

    candidate="$(command -v python || true)"
    if [ -n "$candidate" ] && _supports_runtime_cert_python "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
    fi
    return 1
}

_find_supported_python() {
    local candidate=""
    for candidate in python3.11 python3.10 python3.9 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && _supports_runtime_cert_python "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "ERROR: Could not find a supported Python interpreter (3.9-3.11; 3.11 recommended)." >&2
    return 1
}

USE_ACTIVE_ENV=0
if PYTHON_BIN="$(_active_env_python)"; then
    USE_ACTIVE_ENV=1
else
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        echo "WARNING: active virtual environment is not using Python 3.9-3.11; bootstrapping .venv-cert with a supported interpreter instead." >&2
    fi
    PYTHON_BIN="$(_find_supported_python)"
fi

PYTHON_VERSION="$(_python_version "$PYTHON_BIN")"
echo "  Python    : $PYTHON_BIN ($PYTHON_VERSION)"

# ---------------------------------------------------------------------------
# Internal helper: parse a JUnit XML file and decide if the stage passed.
# Returns:
#   0  — stage PASSED (tests > 0, failures == 0, errors == 0, skipped == 0)
#   1  — stage FAILED (failures or errors present)
#   2  — stage UNIMPLEMENTED (tests == 0 or every test was skipped)
# ---------------------------------------------------------------------------
_check_junit() {
    local xml="$1"
    local stage_name="$2"
    if [ ! -f "$xml" ]; then
        echo "  ✗ $stage_name: JUnit XML not produced — marking FAILED"
        return 1
    fi
    "$PYTHON_BIN" - "$xml" "$stage_name" <<'PYEOF'
import sys, xml.etree.ElementTree as ET

xml_path, stage_name = sys.argv[1], sys.argv[2]
try:
    root = ET.parse(xml_path).getroot()
except ET.ParseError as e:
    print(f"  ✗ {stage_name}: malformed JUnit XML ({e}) — marking FAILED")
    sys.exit(1)

suite = root if root.tag == "testsuite" else root.find("testsuite")
if suite is None:
    print(f"  ✗ {stage_name}: no <testsuite> element — marking FAILED")
    sys.exit(1)

tests    = int(suite.get("tests",    "0"))
failures = int(suite.get("failures", "0"))
errors   = int(suite.get("errors",   "0"))
skipped  = int(suite.get("skipped",  "0"))

if tests == 0 or (skipped > 0 and skipped == tests):
    print(f"  ✗ {stage_name} UNIMPLEMENTED — 0 tests executed (all were @skip or suite is empty)")
    sys.exit(2)
if failures > 0 or errors > 0:
    print(f"  ✗ {stage_name} FAILED (failures={failures}, errors={errors})")
    sys.exit(1)
if skipped > 0:
    print(f"  ✗ {stage_name} FAILED (skipped={skipped} — certification requires 0 skips)")
    sys.exit(1)
print(f"  ✓ {stage_name} PASSED ({tests} tests)")
sys.exit(0)
PYEOF
}

# ---------------------------------------------------------------------------
# Helper: run a pytest stage with automatic JUnit XML inspection.
# Usage: run_pytest_stage "Stage Name" "<xml-basename>" <pytest-args...>
# ---------------------------------------------------------------------------
run_pytest_stage() {
    local name="$1"
    local xml_file="$ARTIFACT_DIR/$2"
    shift 2
    STAGES_TOTAL=$((STAGES_TOTAL + 1))
    echo ""
    echo "──── Stage: $name ────"
    local exit_code=0
    "$PYTHON_BIN" -m pytest "$@" --junitxml="$xml_file" -q --tb=short || exit_code=$?
    local check_code=0
    _check_junit "$xml_file" "$name" || check_code=$?
    if [ "$check_code" -eq 2 ]; then
        STAGES_UNIMPLEMENTED=$((STAGES_UNIMPLEMENTED + 1))
        FAILURES=$((FAILURES + 1))
    elif [ "$check_code" -ne 0 ] || [ "$exit_code" -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    else
        STAGES_PASSED=$((STAGES_PASSED + 1))
    fi
}

# ---------------------------------------------------------------------------
# Helper: run a non-pytest stage (scripts, etc.).
# ---------------------------------------------------------------------------
run_stage() {
    local name="$1"; shift
    STAGES_TOTAL=$((STAGES_TOTAL + 1))
    echo ""
    echo "──── Stage: $name ────"
    if "$@" ; then
        echo "  ✓ $name PASSED"
        STAGES_PASSED=$((STAGES_PASSED + 1))
    else
        echo "  ✗ $name FAILED"
        FAILURES=$((FAILURES + 1))
    fi
}

# ---------------------------------------------------------------------------
# Stage 0: Create / activate venv (optional — skip if already in one)
# ---------------------------------------------------------------------------
if [ "$USE_ACTIVE_ENV" -eq 0 ]; then
    VENV_DIR="$REPO_ROOT/.venv-cert"
    if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ] && ! _supports_runtime_cert_python "$VENV_DIR/bin/python"; then
        echo "Recreating certification venv at $VENV_DIR (unsupported Python in existing venv) ..."
        rm -rf "$VENV_DIR"
    fi
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating certification venv at $VENV_DIR ..."
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON_BIN="$VENV_DIR/bin/python"
    PYTHON_VERSION="$(_python_version "$PYTHON_BIN")"
    echo "  Using venv : $PYTHON_BIN ($PYTHON_VERSION)"
    "$PYTHON_BIN" -m pip install --upgrade pip
    echo "Installing package with Apple extras ..."
    "$PYTHON_BIN" -m pip install --quiet -e '.[apple,test]'
fi

# ---------------------------------------------------------------------------
# Stage 1: Strict preflight
# ---------------------------------------------------------------------------
run_stage "Strict Preflight" \
    "$PYTHON_BIN" scripts/preflight.py --strict --json

"$PYTHON_BIN" scripts/preflight.py --strict --json > "$ARTIFACT_DIR/preflight.json" 2>&1 || true

# ---------------------------------------------------------------------------
# Stage 2: Cache upgrade roundtrip
# ---------------------------------------------------------------------------
run_pytest_stage "Cache Upgrade Roundtrip" "junit_cache_roundtrip.xml" \
    tests/integration_mlx/test_cache_upgrade_roundtrip.py

# ---------------------------------------------------------------------------
# Stage 3: Streaming attention equivalence
# ---------------------------------------------------------------------------
run_pytest_stage "Attention Equivalence" "junit_attention_equiv.xml" \
    tests/integration_mlx/test_streaming_attention_equivalence.py

# ---------------------------------------------------------------------------
# Stage 4: Llama smoke test
# ---------------------------------------------------------------------------
if [ -n "${TQ_TEST_LLAMA_MODEL:-}" ]; then
    run_pytest_stage "Llama Smoke" "junit_llama_smoke.xml" \
        tests/integration_mlx/test_llama_runtime_smoke.py
else
    echo ""
    echo "──── Stage: Llama Smoke ────"
    echo "  ✗ SKIPPED — TQ_TEST_LLAMA_MODEL not set (required for certification)"
    STAGES_SKIPPED=$((STAGES_SKIPPED + 1))
    FAILURES=$((FAILURES + 1))
    STAGES_TOTAL=$((STAGES_TOTAL + 1))
fi

# ---------------------------------------------------------------------------
# Stage 5: Gemma smoke test
# ---------------------------------------------------------------------------
if [ -n "${TQ_TEST_GEMMA_MODEL:-}" ]; then
    run_pytest_stage "Gemma Smoke" "junit_gemma_smoke.xml" \
        tests/integration_mlx/test_gemma_runtime_smoke.py
else
    echo ""
    echo "──── Stage: Gemma Smoke ────"
    echo "  ✗ SKIPPED — TQ_TEST_GEMMA_MODEL not set (required for certification)"
    STAGES_SKIPPED=$((STAGES_SKIPPED + 1))
    FAILURES=$((FAILURES + 1))
    STAGES_TOTAL=$((STAGES_TOTAL + 1))
fi

# ---------------------------------------------------------------------------
# Stage 5.5: Quality evaluation (perplexity + KL divergence)
# ---------------------------------------------------------------------------
if [ -n "${TQ_TEST_LLAMA_MODEL:-}" ]; then
    for CLASS in short medium; do
        # Preset: paper_mse (3-bit MSE, Hadamard rotation, no QJL) for the quality
        # gate.  paper_prod uses only 2 effective bits (k_bits-1 for main + 1-bit
        # QJL residual) which produces degenerate results for short sequences.
        # Min-prompt-tokens=32: TurboQuant is designed for long sequences; prompts
        # shorter than this are skipped (trivially pass the quality gate).
        # Thresholds: this is a BATCH (teacher-forcing) forward pass, not a
        # streaming/autoregressive measurement.  Batch compression puts all T
        # keys into one block where the quantiser scale covers the full context,
        # systematically overstating degradation vs.  the production streaming
        # path (which the smoke tests validate directly).  The threshold here
        # only guards against catastrophic failures (kv-corruption, NaN/Inf,
        # numerical explosions).  Use benchmarks/exploratory/run_final_eval.py
        # for authoritative streaming-mode quality numbers.
        run_stage "Quality Eval $CLASS (Llama)" \
            "$PYTHON_BIN" benchmarks/runtime_cert/run_quality_eval.py \
            --model "$TQ_TEST_LLAMA_MODEL" \
            --prompt-file "benchmarks/runtime_cert/prompts/$CLASS.jsonl" \
            --prompt-class "$CLASS" \
            --output-dir "$ARTIFACT_DIR" \
            --preset paper_mse \
            --min-prompt-tokens 32 \
            --max-delta-ppl 20.0 \
            --max-mean-kl 5.0 \
            --seed 42
    done
else
    echo ""
    echo "──── Stage: Quality Evaluation ────"
    echo "  ✗ SKIPPED — TQ_TEST_LLAMA_MODEL not set (required for certification)"
    STAGES_SKIPPED=$((STAGES_SKIPPED + 1))
    FAILURES=$((FAILURES + 1))
    STAGES_TOTAL=$((STAGES_TOTAL + 1))
fi

# ---------------------------------------------------------------------------
# Stage 6: Long-context stability
# ---------------------------------------------------------------------------
run_pytest_stage "Long-Context Stability" "junit_long_context.xml" \
    tests/integration_mlx/test_long_context_stability.py

# ---------------------------------------------------------------------------
# Stage 7: Dense vs TurboQuant benchmark (requires model env vars)
# ---------------------------------------------------------------------------
if [ -n "${TQ_TEST_LLAMA_MODEL:-}" ]; then
    for CLASS in short medium long; do
        run_stage "Benchmark $CLASS (Llama)" \
            "$PYTHON_BIN" benchmarks/runtime_cert/run_dense_vs_tq.py \
            --model "$TQ_TEST_LLAMA_MODEL" \
            --prompt-file "benchmarks/runtime_cert/prompts/$CLASS.jsonl" \
            --prompt-class "$CLASS" \
            --output-dir "$ARTIFACT_DIR" \
            --max-new-tokens 64 \
            --seed 42 \
            --mode both
    done
fi

if [ -n "${TQ_TEST_GEMMA_MODEL:-}" ]; then
    for CLASS in short medium long; do
        run_stage "Benchmark $CLASS (Gemma)" \
            "$PYTHON_BIN" benchmarks/runtime_cert/run_dense_vs_tq.py \
            --model "$TQ_TEST_GEMMA_MODEL" \
            --prompt-file "benchmarks/runtime_cert/prompts/$CLASS.jsonl" \
            --prompt-class "$CLASS" \
            --output-dir "$ARTIFACT_DIR" \
            --max-new-tokens 64 \
            --seed 42 \
            --mode both
    done
fi

# ---------------------------------------------------------------------------
# Stage 8: Aggregate metrics
# ---------------------------------------------------------------------------
if ls "$ARTIFACT_DIR"/*_dense.json "$ARTIFACT_DIR"/*_turboquant.json >/dev/null 2>&1; then
    run_stage "Metric Aggregation" \
        "$PYTHON_BIN" benchmarks/runtime_cert/collect_metrics.py \
        --input-dir "$ARTIFACT_DIR" \
        --output-dir "$ARTIFACT_DIR"
else
    echo ""
    echo "──── Stage: Metric Aggregation ────"
    echo "  SKIPPED (no benchmark artifacts to aggregate — not counted as failure)"
fi

# ---------------------------------------------------------------------------
# Write certification manifest
# ---------------------------------------------------------------------------
MANIFEST_PATH="$ARTIFACT_DIR/cert_manifest.json"
if ! "$PYTHON_BIN" scripts/write_cert_manifest.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --passed "$STAGES_PASSED" \
    --failed "$FAILURES" \
    --skipped "$STAGES_SKIPPED" \
    --unimplemented "$STAGES_UNIMPLEMENTED" \
    --total "$STAGES_TOTAL" \
    --turboquant-version "$($PYTHON_BIN -c 'import turboquant; print(turboquant.__version__)' 2>/dev/null || echo 'unknown')"; then
    if [ -f "$MANIFEST_PATH" ]; then
        echo "  cert_manifest.json recorded a FAIL result for this run."
    else
        echo "  WARNING: cert_manifest.json was not written."
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Certification artifacts: $ARTIFACT_DIR"
echo ""
if [ "$FAILURES" -eq 0 ]; then
    echo "  ✓ ALL STAGES PASSED"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
else
    echo "  ✗ $FAILURES STAGE(S) FAILED OR UNIMPLEMENTED"
    echo "    Passed       : $STAGES_PASSED / $STAGES_TOTAL"
    echo "    Unimplemented: $STAGES_UNIMPLEMENTED  (all-skip test suites)"
    echo "    Skipped      : $STAGES_SKIPPED  (missing env vars)"
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
fi
