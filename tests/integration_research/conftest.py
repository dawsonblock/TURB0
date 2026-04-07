"""
pytest conftest for the integration_research suite.

These tests run research benchmark scripts through subprocesses so the pytest
process itself does not import MLX or hold on to Metal state. Keep this suite
isolated from tests/unit_static and tests/integration_mlx just like the rest of
the repo's MLX-sensitive lanes.
"""
