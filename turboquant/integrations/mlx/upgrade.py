"""
turboquant.integrations.mlx.upgrade — production KV-cache upgrade policy.

This module owns the policy for when and how to promote a dense KVCache to a
TurboQuantKCache.  It is intentionally separate from ``generate.py`` so that
the policy can be unit-tested, reused across frontends, and evolved without
touching the main generation loop.

The legacy helper ``maybe_turboquant_k_cache`` in ``generate.py`` now
delegates to :func:`upgrade_cache_list`.

Usage
-----
    from turboquant.config import TurboQuantConfig
    from turboquant.integrations.mlx.upgrade import upgrade_cache_list
    from turboquant.runtime.events import (
        EventLog,
        record_runtime_upgrade_events,
    )

    config = TurboQuantConfig(k_bits=3, k_group_size=64, ...)
    events = upgrade_cache_list(
        prompt_cache,
        k_start=512,
        config=config,
        model_family="llama",
    )
    for ev in events:
        if ev.upgraded:
            print(f"layer {ev.layer_index}: {ev.old_type} → {ev.new_type} "
                  f"at offset {ev.offset_at_upgrade}")

    # Optional persistence bridge for certification / instrumentation flows.
    log = EventLog(artifact_dir=None)
    record_runtime_upgrade_events(log, events)
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as _dc_replace

from turboquant.config import TurboQuantConfig
from turboquant.runtime.support import assert_supported_model_family

# -- Event --------------------------------------------------------------------


@dataclass
class CacheUpgradeEvent:
    """Record of a single cache-layer upgrade decision.

    This is the **primary runtime event type** returned by
    :func:`upgrade_cache_list`.  It is a lightweight dataclass with no
    persistence logic.

    For JSONL artifact persistence (e.g. writing ``events.jsonl`` during
    certification runs), use :class:`turboquant.runtime.events.EventLog`
    together with its own ``CacheUpgradeEvent`` or the explicit
    ``record_runtime_upgrade_events(...)`` adapter. The canonical runtime
    path returns these lightweight in-process decision events.
    It does not automatically persist them.

    Fields
    ------
    upgraded:
        ``True`` if the layer was promoted to TurboQuantKCache.
    layer_index:
        Zero-based index of the layer in *prompt_cache*.
    old_type:
        ``type(cache).__name__`` before the upgrade (or the same type if
        no upgrade occurred).
    new_type:
        ``type(cache).__name__`` after the upgrade.
    offset_at_upgrade:
        ``cache.offset`` at the moment the decision was made.
    old_bytes:
        Approximate byte footprint of the dense cache before upgrade.
        Set to 0 when ``upgraded=False``.
    new_bytes:
        Approximate byte footprint of the TurboQuant cache after upgrade.
        Set to 0 when ``upgraded=False``.
    ratio:
        ``new_bytes / old_bytes`` — compression ratio (<1.0 is smaller).
        0.0 when ``upgraded=False`` or ``old_bytes=0``.
    """

    upgraded: bool
    layer_index: int
    old_type: str
    new_type: str
    offset_at_upgrade: int
    old_bytes: int = 0
    new_bytes: int = 0
    ratio: float = 0.0


# -- Upgrade policy -----------------------------------------------------------


def upgrade_cache_list(
    prompt_cache: list,
    k_start: int | None,
    config: TurboQuantConfig,
    model_family: str | None = None,
) -> list[CacheUpgradeEvent]:
    """Promote KVCache entries to TurboQuantKCache when their offset threshold
    is reached.

    This is the canonical upgrade path used by the mlx-lm generation loop.
    Call once per generation step; the function is idempotent — layers that
    have already been upgraded are skipped.

    Parameters
    ----------
    prompt_cache:
        The per-layer cache list.  Modified in place when an upgrade occurs.
    k_start:
        Minimum ``cache.offset`` before upgrading.  ``None`` disables all
        upgrades (every layer stays as-is).
    config:
        :class:`turboquant.config.TurboQuantConfig` governing compression.
        The production path always uses ``return_mode="view"``; the legacy
        ``return_mode`` kwarg is not surfaced here.
    model_family:
        Model architecture family (e.g. ``"llama"`` or ``"gemma"``).
        Must be in the supported allowlist or
        :class:`~turboquant.errors.UnsupportedModelError` is raised before
        any cache is mutated.  ``None`` is never valid; callers that cannot
        determine the family should skip the upgrade entirely.

    Returns
    -------
    List[CacheUpgradeEvent]
        One event per cache layer, in order.  Inspect ``ev.upgraded`` to
        see which layers were promoted this call.
    """
    # Gate 2 — model allowlist.  Must be checked before any cache mutation.
    # Fail-closed: None is not a valid bypass; callers must supply a family
    # string from SUPPORTED_FAMILIES.  Use maybe_turboquant_k_cache / the
    # generate_step shim (which skips the upgrade when inference returns None)
    # rather than passing None here.
    if model_family is None:
        from turboquant.errors import UnsupportedModelError

        raise UnsupportedModelError(
            "model_family must be specified; pass 'llama' or 'gemma'. "
            "Got None — unknown or unsupported model architecture."
        )
    assert_supported_model_family(model_family)

    # Lazy import to avoid circular deps and to keep this module importable
    # even if turboquant or mlx_lm is not fully initialised.
    from turboquant.integrations.mlx._cache_adapter import TurboQuantKCache

    events: list[CacheUpgradeEvent] = []

    if k_start is None:
        # Fast path: no upgrade policy in effect.
        for i, c in enumerate(prompt_cache):
            events.append(
                CacheUpgradeEvent(
                    upgraded=False,
                    layer_index=i,
                    old_type=type(c).__name__,
                    new_type=type(c).__name__,
                    offset_at_upgrade=getattr(c, "offset", 0),
                )
            )
        return events

    for i, c in enumerate(prompt_cache):
        old_type = type(c).__name__
        cur_offset = getattr(c, "offset", 0)

        # Already upgraded — skip.
        if isinstance(c, TurboQuantKCache):
            events.append(
                CacheUpgradeEvent(
                    upgraded=False,
                    layer_index=i,
                    old_type=old_type,
                    new_type=old_type,
                    offset_at_upgrade=cur_offset,
                )
            )
            continue

        # Threshold not yet reached or missing required properties to extract
        # keys/values.
        if cur_offset < k_start or not hasattr(c, "keys") or not hasattr(c, "values"):
            events.append(
                CacheUpgradeEvent(
                    upgraded=False,
                    layer_index=i,
                    old_type=old_type,
                    new_type=old_type,
                    offset_at_upgrade=cur_offset,
                )
            )
            continue

        # Canonical upgrade path: clone the caller's config, overriding only
        # return_mode for production safety.  Using dataclasses.replace
        # preserves all fields including residual_mode, qjl_bits, etc.
        prod_config = _dc_replace(config, return_mode="view")

        tq = TurboQuantKCache(prod_config)

        # Perform the actual data transfer.
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        if keys is not None and values is not None:
            tq.update_and_fetch(keys[..., :cur_offset, :], values[..., :cur_offset, :])

        prompt_cache[i] = tq

        # Compute byte footprints for the event record.
        def get_bytes(cache_obj):
            if hasattr(cache_obj, "byte_size"):
                return cache_obj.byte_size()
            return 0

        old_b = get_bytes(c)
        new_b = get_bytes(tq)

        events.append(
            CacheUpgradeEvent(
                upgraded=True,
                layer_index=i,
                old_type=old_type,
                new_type="TurboQuantKCache",
                offset_at_upgrade=cur_offset,
                old_bytes=old_b,
                new_bytes=new_b,
                ratio=new_b / old_b if old_b > 0 else 0.0,
            )
        )
    return events
