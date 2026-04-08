from __future__ import annotations

from pathlib import Path

import mlx.core as mx

_KERNELS: dict[tuple[int, int], object] = {}


def _source() -> str:
    return (Path(__file__).parent / "topk_residual_decode.metal").read_text()


def decode_topk_residual_metal(
    values: mx.array,
    indices: mx.array,
    *,
    group_size: int,
) -> mx.array:
    k = int(values.shape[-1])
    cache_key = (group_size, k)
    if cache_key not in _KERNELS:
        _KERNELS[cache_key] = mx.fast.metal_kernel(
            name="decode_topk_residual",
            input_names=["values", "indices"],
            output_names=["out"],
            source=_source(),
        )

    kernel = _KERNELS[cache_key]
    *prefix, n_groups, _ = values.shape
    output_shape = (*prefix, n_groups, group_size)
    prefix_size = 1
    for dim in prefix:
        prefix_size *= int(dim)
    total_elements = prefix_size * int(n_groups) * group_size
    out = kernel(
        inputs=[values, indices],
        output_shapes=[output_shape],
        output_dtypes=[values.dtype],
        grid=(total_elements, 1, 1),
        threadgroup=(64, 1, 1),
        template=[
            ("GROUP_SIZE", group_size),
            ("TOPK", k),
            ("N_GROUPS", int(n_groups)),
        ],
        stream=mx.gpu,
    )
    return out[0]
