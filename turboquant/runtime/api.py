from turboquant.config import TurboQuantConfig


class TurboQuantRuntime:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TurboQuantRuntime was an experimental prototype and has been removed from the supported surface. "
            "Use the canonical MLX integration: from turboquant.integrations.mlx.cache_adapter import TurboQuantKCache"
        )


    def step(self, keys, values):
        pass

    def attention(self, queries, state):
        pass
