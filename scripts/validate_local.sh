#!/usr/bin/env bash
set -e

echo "Running TurboQuant Local Validation..."

# 1. Preflight
python3 scripts/preflight.py

# 2. Run static tests
printf '\nRunning static tests...\n'
python3 -m pytest tests/unit_static/

# 3. Optional: Run strictly hardware dependent tests if on platform
runtime_validation_ran=false
if python3 scripts/preflight.py --strict >/dev/null 2>&1; then
    runtime_validation_ran=true
    printf '\nRunning MLX-dependent tests...\n'
    python3 -m pytest tests/unit_mlx/
    python3 -m pytest tests/integration_mlx/
else
    printf '\nSkipping MLX tests (not on Apple Silicon / missing MLX).\n'
fi

if [ "$runtime_validation_ran" = true ]; then
    printf '\nStatic/local validation passed.\n'
    printf 'Full Apple runtime validation passed.\n'
else
    printf '\nStatic/local validation passed.\n'
    printf 'Full Apple runtime validation was skipped because strict preflight requirements were not met.\n'
fi
