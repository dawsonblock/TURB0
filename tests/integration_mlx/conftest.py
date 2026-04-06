"""
pytest conftest for the integration_mlx suite.

ISOLATION NOTE
--------------
These tests use the MLX Metal GPU directly.  Running them in the same
pytest process as tests/unit_static (which manipulates sys.modules["mlx"])
can corrupt the Metal device state and cause C-level aborts.

For a clean run, invoke the suites in separate processes:

    python -m pytest tests/unit_static          # static checks, no Metal
    python -m pytest tests/unit_mlx             # MLX unit tests
    python -m pytest tests/integration_mlx      # MLX integration tests

The Makefile targets handle this correctly:
    make test-static
    make test-unit-mlx
    make test-structural

Combining them in one invocation (e.g. pytest tests/unit_static tests/integration_mlx)
is known to abort due to Metal GPU state isolation requirements.
"""
# No fixtures needed — isolation is enforced at the invocation level.
