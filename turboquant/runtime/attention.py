from __future__ import annotations

import math

import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import EncodedKeyBlock
from turboquant.core.residual_codec import ResidualPayload, build_residual_codec


def _rotate_queries(
    q: mx.array,
    *,
    config: TurboQuantConfig,
    orig_dim: int,
) -> mx.array:
    from turboquant.core.rotation import FixedRotation

    rotation = FixedRotation.from_config(config, orig_dim)
    return rotation.apply(q.astype(mx.float32)).astype(q.dtype)


def _repeat_kv_heads(x: mx.array, target_heads: int) -> mx.array:
    if x.shape[-3] == target_heads:
        return x
    n_rep = target_heads // x.shape[-3]
    return mx.repeat(x, n_rep, axis=-3)


def _apply_softcap_and_mask(
    scores: mx.array,
    *,
    softcap,
    mask,
    q_len: int,
    total_k_len: int,
    chunk_start: int,
    chunk_stop: int,
) -> mx.array:
    if softcap is not None:
        scores = mx.tanh(scores / softcap) * softcap

    if mask == "causal":
        if q_len > 1:
            inds = mx.arange(chunk_start, chunk_stop)[None, None, :]
            q_inds = mx.arange(total_k_len - q_len, total_k_len)[None, :, None]
            scores = mx.where(
                inds > q_inds,
                mx.array(-1e9, dtype=scores.dtype),
                scores,
            )
        return scores

    if mask is None:
        return scores

    chunk_mask = mask[..., chunk_start:chunk_stop]
    if chunk_mask.dtype == mx.bool_:
        return mx.where(chunk_mask, scores, mx.array(-1e9, dtype=scores.dtype))
    return scores + chunk_mask


def _online_softmax_update(
    acc: mx.array | None,
    denom: mx.array | None,
    max_score: mx.array | None,
    scores: mx.array,
    values: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    chunk_max = mx.max(scores, axis=-1, keepdims=True)
    chunk_weights = mx.exp(scores - chunk_max)
    chunk_denom = mx.sum(chunk_weights, axis=-1, keepdims=True)
    chunk_acc = chunk_weights @ values

    if acc is None or denom is None or max_score is None:
        return chunk_acc, chunk_denom, chunk_max

    new_max = mx.maximum(max_score, chunk_max)
    prev_scale = mx.exp(max_score - new_max)
    chunk_scale = mx.exp(chunk_max - new_max)
    return (
        prev_scale * acc + chunk_scale * chunk_acc,
        prev_scale * denom + chunk_scale * chunk_denom,
        new_max,
    )


def _decode_values_slice(impl, start: int, stop: int) -> mx.array:
    if getattr(impl, "storage_mode", "k_only") == "paper_kv" and impl._v_packed_flat is not None:
        assert impl._v_config is not None
        assert impl._v_dequantize is not None
        return impl._v_dequantize(
            impl._v_packed_flat[..., start:stop, :],
            impl._v_scales_flat[..., start:stop, :],
            config=impl._v_config,
        )
    if impl._v_dense_flat is not None:
        return impl._v_dense_flat[..., start:stop, :]
    vals = mx.concatenate(impl.v_cache, axis=-2)
    return vals[..., start:stop, :]


def _main_scores_from_flat_slice(
    q_rot: mx.array,
    *,
    packed_main: mx.array,
    scales: mx.array,
    config: TurboQuantConfig,
    dequantize_main,
    d_rot: int,
    q_heads: int,
) -> mx.array:
    main_hat = dequantize_main(packed_main, scales, config=config)
    main_rot = _repeat_kv_heads(main_hat[..., :d_rot], q_heads)
    return q_rot @ mx.swapaxes(main_rot, -1, -2)


def _qjl_scores_from_flat_slice(
    q_rot: mx.array,
    *,
    bits: mx.array,
    norms: mx.array,
    meta: dict,
    config: TurboQuantConfig,
) -> mx.array:
    codec = build_residual_codec(config)
    payload = ResidualPayload(mode="qjl", data={"bits": bits, "norms": norms, "meta": meta})
    return codec.dot_estimate(q_rot, payload, config=config)


def score_block(
    q: mx.array,
    block: EncodedKeyBlock,
    *,
    config: TurboQuantConfig,
    dequantize_main,
) -> mx.array:
    """
    Compute attention scores against one encoded key block.

    q:
        [..., q_len, d_head]  (original coordinate space)

    Returns:
        [..., q_len, k_len]

    K inside the block is stored in rotated space.  This function rotates
    *q* internally to match.
    """
    config.validate()

    orig_dim = block.orig_dim if block.orig_dim > 0 else block.d_head
    q_rot = _rotate_queries(q, config=config, orig_dim=orig_dim)

    if block.polar is not None:
        main_hat = dequantize_main(block.polar, None, config=config)
    else:
        main_hat = dequantize_main(
            block.packed_main,
            block.scales,
            config=config,
        )
    main_rot = main_hat[..., : block.d_rot]

    if int(q_rot.shape[-1]) != int(main_rot.shape[-1]):
        raise ValueError(
            "q_rot dim "
            f"{int(q_rot.shape[-1])} != main_rot dim "
            f"{int(main_rot.shape[-1])}"
        )

    main_rot = _repeat_kv_heads(main_rot, int(q_rot.shape[-3]))
    main_scores = q_rot @ mx.swapaxes(main_rot, -1, -2)

    codec = build_residual_codec(config)

    if block.residual.mode == "none":
        return main_scores

    if block.residual.mode == "topk":
        resid_hat = codec.decode(block.residual, config=config)
        assert resid_hat is not None
        resid_rot = _repeat_kv_heads(resid_hat[..., : block.d_rot], int(q_rot.shape[-3]))
        return main_scores + q_rot @ mx.swapaxes(resid_rot, -1, -2)

    if block.residual.mode == "qjl":
        resid_scores = codec.dot_estimate(q_rot, block.residual, config=config)
        if tuple(resid_scores.shape) != tuple(main_scores.shape):
            raise ValueError(
                f"QJL residual score shape {tuple(resid_scores.shape)} "
                f"!= main score shape {tuple(main_scores.shape)}"
            )
        return main_scores + resid_scores

    raise ValueError(f"Unsupported residual mode: {block.residual.mode}")


def _legacy_streaming_scores(
    q: mx.array,
    *,
    cache,
    config: TurboQuantConfig,
    dequantize_main,
) -> list[mx.array]:
    out: list[mx.array] = []
    for block in cache.iter_blocks():
        out.append(
            score_block(
                q,
                block,
                config=config,
                dequantize_main=dequantize_main,
            )
        )
    return out


def streaming_scores(
    q: mx.array,
    *,
    cache,
    config: TurboQuantConfig,
    dequantize_main,
) -> list[mx.array]:
    """
    Produce per-block score tensors (q in original coordinate space).
    """
    return _legacy_streaming_scores(
        q,
        cache=cache,
        config=config,
        dequantize_main=dequantize_main,
    )


def _fast_streaming_attention(
    queries,
    impl,
    *,
    scale=1.0,
    mask=None,
    softcap=None,
):
    config = impl.config
    q_scaled = queries * scale
    q_rot = _rotate_queries(
        q_scaled,
        config=config,
        orig_dim=impl._runtime_orig_dim or int(queries.shape[-1]),
    )

    acc = None
    denom = None
    max_score = None
    total_k_len = int(impl._offset)
    q_len = int(queries.shape[-2])
    q_heads = int(queries.shape[-3])

    for chunk in impl.iter_runtime_chunks():
        main_scores = _main_scores_from_flat_slice(
            q_rot,
            packed_main=impl._k_packed_flat[..., chunk.start : chunk.stop, :],
            scales=impl._k_scales_flat[..., chunk.start : chunk.stop, :],
            config=config,
            dequantize_main=impl.dequantize_main,
            d_rot=impl._runtime_d_rot,
            q_heads=q_heads,
        )

        if config.is_prod_mode():
            main_scores = main_scores + _qjl_scores_from_flat_slice(
                q_rot,
                bits=impl._k_qjl_bits_flat[..., chunk.start : chunk.stop, :],
                norms=impl._k_qjl_norms_flat[..., chunk.start : chunk.stop, :],
                meta=impl._k_qjl_meta or {},
                config=config,
            )

        scores = _apply_softcap_and_mask(
            main_scores,
            softcap=softcap,
            mask=mask,
            q_len=q_len,
            total_k_len=total_k_len,
            chunk_start=chunk.start,
            chunk_stop=chunk.stop,
        )
        values = _repeat_kv_heads(_decode_values_slice(impl, chunk.start, chunk.stop), q_heads)
        acc, denom, max_score = _online_softmax_update(
            acc, denom, max_score, scores, values
        )

    if acc is None or denom is None:
        raise ValueError("TurboQuant cache has no runtime chunks to attend over.")

    return acc / mx.maximum(denom, mx.array(math.exp(-20), dtype=denom.dtype))


def _legacy_streaming_attention(
    queries,
    keys_view,
    *,
    scale=1.0,
    mask=None,
    softcap=None,
):
    cache = keys_view.cache
    impl = getattr(cache, "_impl", cache)
    config = impl.config
    dequantize_main = impl.dequantize_main

    scores = _legacy_streaming_scores(
        queries * scale,
        cache=impl,
        config=config,
        dequantize_main=dequantize_main,
    )
    scores = mx.concatenate(scores, axis=-1)

    if softcap is not None:
        scores = mx.tanh(scores / softcap) * softcap

    if mask == "causal":
        q_len = queries.shape[-2]
        k_len = int(scores.shape[-1])
        if q_len > 1:
            inds = mx.arange(k_len)[None, None, :]
            q_inds = mx.arange(k_len - q_len, k_len)[None, :, None]
            mask = mx.where(
                inds > q_inds,
                mx.array(-1e9, dtype=scores.dtype),
                mx.array(0.0, dtype=scores.dtype),
            )
        else:
            mask = None
    if mask is not None:
        scores = scores + mask

    if getattr(impl, "storage_mode", "k_only") == "paper_kv" and impl.v_blocks:
        vals = mx.concatenate(
            [impl.decode_v_block(i) for i in range(len(impl.v_blocks))],
            axis=-2,
        )
    else:
        vals = mx.concatenate(cache.v_cache, axis=-2)

    attn = mx.softmax(scores, axis=-1)
    vals = _repeat_kv_heads(vals, int(queries.shape[-3]))
    return attn @ vals


def turboquant_streaming_attention(
    queries, keys_view, scale=1.0, mask=None, softcap=None
):
    """Compute streaming attention against a TurboQuantKeysView.

    Parameters
    ----------
    softcap:
        Optional logit soft-capping scalar (used by Gemma 2).  When provided,
        scores are transformed as ``tanh(scores / softcap) * softcap`` before
        masking and online-softmax accumulation are applied.
    """
    cache = keys_view.cache
    impl = getattr(cache, "_impl", cache)

    if impl.runtime_fastpath_supported():
        return _fast_streaming_attention(
            queries,
            impl,
            scale=scale,
            mask=mask,
            softcap=softcap,
        )
    return _legacy_streaming_attention(
        queries,
        keys_view,
        scale=scale,
        mask=mask,
        softcap=softcap,
    )
