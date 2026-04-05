

class TurboQuantRuntime:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TurboQuantRuntime was an experimental prototype and has been removed from the supported surface. "
            "Do not instantiate TurboQuantKCache directly; use the canonical MLX integration and upgrade dense prompt caches with "
            "turboquant.integrations.mlx.upgrade.upgrade_cache_list(...)"
        )
