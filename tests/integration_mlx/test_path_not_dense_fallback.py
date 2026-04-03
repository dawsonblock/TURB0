import pytest


@pytest.mark.skip(
    reason="Not implemented: requires Apple Silicon + MLX + model weights. "
    "Must verify TurboQuantKeysView is returned by update_and_fetch, not a dense array."
)
def test_path_not_dense_fallback():
    raise NotImplementedError
