from __future__ import annotations

import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import EncodedKeyBlock
from turboquant.core.residual_codec import build_residual_codec


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

    from turboquant.core.rotation import FixedRotation
    orig_dim = block.orig_dim if block.orig_dim > 0 else block.d_head
    rotation = FixedRotation.from_config(config, orig_dim)
    q_rot = rotation.apply(q.astype(mx.float32)).astype(q.dtype)

    main_hat = dequantize_main(block.packed_main, block.scales, config=config)
    main_rot = main_hat[..., : block.d_rot]

    if int(q_rot.shape[-1]) != int(main_rot.shape[-1]):
        raise ValueError(
            f"q_rot dim {int(q_rot.shape[-1])} != main_rot dim {int(main_rot.shape[-1])}"
        )


    if q_rot.shape[-3] != main_rot.shape[-3]:
        n_rep = q_rot.shape[-3] // main_rot.shape[-3]
        main_rot = mx.repeat(main_rot, n_rep, axis=-3)

    main_scores = q_rot @ mx.swapaxes(main_rot, -1, -2)




    codec = build_residual_codec(config)

    if block.residual.mode == "none":
        return main_scores

    if block.residual.mode == "topk":
        resid_hat = codec.decode(block.residual, config=config)
        resid_rot = resid_hat[..., : block.d_rot]

        if q_rot.shape[-3] != resid_rot.shape[-3]:
            resid_rot = mx.repeat(resid_rot, q_rot.shape[-3] // resid_rot.shape[-3], axis=-3)

        resid_scores = q_rot @ mx.swapaxes(resid_rot, -1, -2)
        return main_scores + resid_scores




    if block.residual.mode == "qjl":
        """
        QJL path:
        dot_estimate() is expected to return residual score contribution
        in [..., q_len, k_len] form.
        """
        resid_scores = codec.dot_estimate(q_rot, block.residual, config=config)
        if tuple(resid_scores.shape) != tuple(main_scores.shape):
            raise ValueError(
                f"QJL residual score shape {tuple(resid_scores.shape)} "
                f"!= main score shape {tuple(main_scores.shape)}"
            )
        return main_scores + resid_scores

    raise ValueError(f"Unsupported residual mode: {block.residual.mode}")

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
    out: list[mx.array] = []
    for block in cache.iter_blocks():
        scores = score_block(
            q,
            block,
            config=config,
            dequantize_main=dequantize_main,
        )
        out.append(scores)
    return out

# Legacy compatibility shim for MLX integrations (llama, gemma, etc.)
def turboquant_streaming_attention(queries, keys_view, scale=1.0, mask=None, softcap=None):
    """Compute streaming attention against a TurboQuantKeysView.

    Parameters
    ----------
    softcap:
        Optional logit soft-capping scalar (used by Gemma 2).  When provided,
        scores are transformed as ``tanh(scores / softcap) * softcap`` before
        the causal mask and softmax are applied.
    """
    cache = keys_view.cache
    import mlx.core as mx

    # Support both TurboQuantKCache (has ._impl) and TurboQuantKVCache (direct).
    impl = getattr(cache, "_impl", cache)
    config = impl.config
    dequantize_main = impl.dequantize_main

    q_scaled = queries * scale

    # compute streaming scores
    scores = streaming_scores(
        q_scaled,
        cache=impl,
        config=config,
        dequantize_main=dequantize_main,
    )

    # We concatenate scores then softmax
    scores = mx.concatenate(scores, axis=-1)

    # Gemma 2-style logit soft-capping (applied before masking).
    if softcap is not None:
        scores = mx.tanh(scores / softcap) * softcap

    if mask == "causal":
        q_len = queries.shape[-2]
        k_len = int(scores.shape[-1])
        if q_len > 1:
            inds = mx.arange(k_len)[None, None, :]
            q_inds = mx.arange(k_len - q_len, k_len)[None, :, None]
            mask = mx.where(inds > q_inds, mx.array(-1e9, dtype=scores.dtype), mx.array(0.0, dtype=scores.dtype))
        else:
            mask = None
    if mask is not None:
        scores = scores + mask


    # Fetch values: paper mode decodes from v_blocks; legacy uses dense v_cache.
    if getattr(impl, "storage_mode", "k_only") == "paper_kv" and impl.v_blocks:
        vals = mx.concatenate(
            [impl.decode_v_block(i) for i in range(len(impl.v_blocks))],
            axis=-2,
        )
    else:
        vals = mx.concatenate(cache.v_cache, axis=-2)


    attn = mx.softmax(scores, axis=-1)
    if queries.shape[-3] != vals.shape[-3]:
        n_rep = queries.shape[-3] // vals.shape[-3]
        vals = mx.repeat(vals, n_rep, axis=-3)
    return attn @ vals

