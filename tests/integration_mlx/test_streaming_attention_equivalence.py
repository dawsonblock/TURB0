import pytest


@pytest.mark.skip(
    reason="Not implemented: requires Apple Silicon + MLX. "
    "Needs turboquant_streaming_attention output compared against dense SDPA "
    "on the same keys/values within tolerance."
)
def test_streaming_attention_equivalence():
    raise NotImplementedError
