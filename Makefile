PYTHON ?= python3
PIP ?= uv pip

.PHONY: help install-dev install-apple compile lint typecheck test test-static test-mlx test-structural test-path-proof test-smoke-llama test-smoke-gemma test-long-context certify-apple-runtime certify-structural build-dist export-source-zip validate-apple clean

help:
	@printf "Targets:\n"
	@printf "  install-dev              Install editable package with dev extras\n"
	@printf "  install-apple            Install editable package with Apple Silicon MLX extras\n"
	@printf "  compile                  Compile all source and test modules\n"
	@printf "  lint                     Run Ruff linting and formatting via Nox\n"
	@printf "  typecheck                Run Mypy type validation via Nox\n"
	@printf "  test                     Alias for test-static (safe on any platform)\n"
	@printf "  test-static              Run static unit tests (no MLX required)\n"
	@printf "  test-mlx                 Run MLX integration tests (Apple Silicon only)\n"
	@printf "  test-structural          Run structural integration tests (no model weights)\n"
	@printf "  test-path-proof          Verify TQ path is exercised, not dense fallback\n"
	@printf "  test-smoke-llama         Llama runtime smoke (TinyModel default; set TQ_TEST_LLAMA_MODEL for real-model mode)\n"
	@printf "  test-smoke-gemma         Gemma runtime smoke (TinyModel default; set TQ_TEST_GEMMA_MODEL for real-model mode)\n"
	@printf "  test-long-context        Long-context stability proof (TinyModel default; set TQ_TEST_LLAMA_MODEL for real-model mode)\n"
	@printf "  certify-apple-runtime    Full Apple-Silicon runtime certification\n"
	@printf "  certify-structural       Structural cert only (no model weights needed)\n"
	@printf "  build-dist               Build wheel and sdist\n"
	@printf "  export-source-zip        Create a clean tracked-files source zip from HEAD or TQ_SOURCE_EXPORT_REF\n"
	@printf "  validate-apple           Run Apple Silicon runtime validation script\n"
	@printf "  clean                    Remove build artifacts\n"

install-dev:
	$(PIP) install -e '.[dev]'

install-apple:
	$(PIP) install -e '.[apple,dev]'

compile:
	$(PYTHON) -m compileall turboquant tests

lint:
	nox -s lint

typecheck:
	nox -s typecheck

test: test-static

test-static:
	nox -s tests_static

test-mlx:
	nox -s tests_mlx

test-unit-mlx:
	$(PYTHON) -m pytest tests/unit_mlx/ -v --tb=short

test-structural:
	$(PYTHON) -m pytest \
		tests/integration_mlx/test_path_not_dense_fallback.py \
		tests/integration_mlx/test_cache_upgrade_roundtrip.py \
		tests/integration_mlx/test_streaming_attention_equivalence.py \
		-v --tb=short

test-path-proof:
	$(PYTHON) -m pytest tests/integration_mlx/test_path_not_dense_fallback.py -v --tb=short

test-smoke-llama:
	$(PYTHON) -m pytest tests/integration_mlx/test_llama_runtime_smoke.py -v --tb=short

test-smoke-gemma:
	$(PYTHON) -m pytest tests/integration_mlx/test_gemma_runtime_smoke.py -v --tb=short

test-long-context:
	$(PYTHON) -m pytest tests/integration_mlx/test_long_context_stability.py -v --tb=short

certify-structural:
	$(PYTHON) -m pytest \
		tests/integration_mlx/test_path_not_dense_fallback.py \
		tests/integration_mlx/test_cache_upgrade_roundtrip.py \
		tests/integration_mlx/test_streaming_attention_equivalence.py \
		-v --tb=short --junitxml=artifacts/junit_structural.xml

certify-apple-runtime:
	./scripts/certify_apple_runtime.sh

build-dist:
	$(PYTHON) -m build

export-source-zip:
	@mkdir -p dist
	@ref="$(TQ_SOURCE_EXPORT_REF)"; \
	if [ -z "$$ref" ]; then ref="HEAD"; fi; \
	safe_ref="$$(printf '%s' "$$ref" | tr '/:' '__')"; \
	out="dist/turboquant-source-$${safe_ref}.zip"; \
	rm -f "$$out"; \
	git archive --format=zip --output="$$out" "$$ref"; \
	printf 'Created clean source export %s from %s\n' "$$out" "$$ref"

validate-apple:
	./scripts/validate_apple_silicon.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .nox .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
