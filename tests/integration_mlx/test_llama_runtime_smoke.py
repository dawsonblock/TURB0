import pytest


@pytest.mark.skip(
    reason="Not implemented: requires Apple Silicon + MLX + Llama model weights "
    "(set TQ_TEST_LLAMA_MODEL env var). Not certified yet."
)
def test_llama_runtime_smoke():
    raise NotImplementedError
