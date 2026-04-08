from __future__ import annotations

import functools
import inspect

_PATCHED = False


def apply_mlx_lm_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        import mlx_lm.generate as generate_mod
        import mlx_lm.models.base as base_mod
        import mlx_lm.models.cache as cache_mod
    except ImportError:
        return

    original_sdpa = getattr(base_mod, "scaled_dot_product_attention", None)
    if callable(original_sdpa) and not getattr(original_sdpa, "_turboquant_patched", False):
        @functools.wraps(original_sdpa)
        def _patched_sdpa(queries, keys, values, cache, scale, mask=None, sinks=None):
            if type(keys).__name__ == "TurboQuantKeysView":
                from turboquant.runtime.attention import turboquant_streaming_attention

                return turboquant_streaming_attention(
                    queries, keys, scale=scale, mask=mask
                )
            return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

        _patched_sdpa._turboquant_patched = True  # type: ignore[attr-defined]
        base_mod.scaled_dot_product_attention = _patched_sdpa

    original_make_prompt_cache = getattr(cache_mod, "make_prompt_cache", None)
    if callable(original_make_prompt_cache) and not getattr(
        original_make_prompt_cache, "_turboquant_patched", False
    ):
        @functools.wraps(original_make_prompt_cache)
        def _patched_make_prompt_cache(model, max_kv_size=None):
            return original_make_prompt_cache(model, max_kv_size=max_kv_size)

        _patched_make_prompt_cache._turboquant_patched = True  # type: ignore[attr-defined]
        cache_mod.make_prompt_cache = _patched_make_prompt_cache

    original_generate_step = getattr(generate_mod, "generate_step", None)
    if callable(original_generate_step) and not getattr(
        original_generate_step, "_turboquant_patched", False
    ):
        original_sig = inspect.signature(original_generate_step)

        @functools.wraps(original_generate_step)
        def _patched_generate_step(*args, **kwargs):
            turboquant_k_start = kwargs.pop("turboquant_k_start", None)
            turboquant_model_family = kwargs.pop("turboquant_model_family", None)
            turboquant_config = kwargs.pop("turboquant_config", None)
            prompt_cache = kwargs.get("prompt_cache")

            if (
                turboquant_k_start is not None
                and prompt_cache is not None
                and turboquant_model_family is not None
                and turboquant_config is not None
            ):
                from turboquant.integrations.mlx.upgrade import upgrade_cache_list

                upgrade_cache_list(
                    prompt_cache,
                    k_start=turboquant_k_start,
                    config=turboquant_config,
                    model_family=turboquant_model_family,
                )

            passthrough = {
                key: value
                for key, value in kwargs.items()
                if key in original_sig.parameters
            }
            return original_generate_step(*args, **passthrough)

        _patched_generate_step._turboquant_patched = True  # type: ignore[attr-defined]
        generate_mod.generate_step = _patched_generate_step

    _PATCHED = True
