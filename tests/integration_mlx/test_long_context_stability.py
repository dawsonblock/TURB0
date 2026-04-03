import pytest


@pytest.mark.skip(
    reason="Not implemented: requires Apple Silicon + MLX + model weights. "
    "Needs long-context (>2048 token) generation with TurboQuantKCache active."
)
def test_long_context_stability():
    raise NotImplementedError
