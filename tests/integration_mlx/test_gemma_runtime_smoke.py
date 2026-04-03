import pytest


@pytest.mark.skip(
    reason="Not implemented: requires Apple Silicon + MLX + Gemma model weights "
    "(set TQ_TEST_GEMMA_MODEL env var). Not certified yet."
)
def test_gemma_runtime_smoke():
    raise NotImplementedError
