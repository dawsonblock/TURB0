import pytest


@pytest.mark.skip(
    reason="Not implemented: requires Apple Silicon + MLX + real KVCache objects. "
    "Implement with TurboQuantKCache + update_and_fetch round-trip once MLX available."
)
def test_cache_upgrade_roundtrip():
    raise NotImplementedError
