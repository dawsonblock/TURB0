import os

import mlx.core as mx

from turboquant.core.pipeline import EncodedKeyBlock, decode_k_block
from turboquant.experimental.kernels.metal.runtime import decode_k_metal


def decode_k_block_metal(block: EncodedKeyBlock, config, d_head: int) -> mx.array:
    """
    Metal-accelerated decode path for specific residual modes.
    Returns the decoded shape or falls back to pure python implementation.
    """
    if os.getenv("TQ_USE_METAL", "0") == "1":
        # Only sparse-topk has a Metal fusion path currently
        if (
            config.residual_mode == "topk"
            and hasattr(block.residual, "data")
            and block.residual.data is not None
        ):
            rv, ri = block.residual.data
            return decode_k_metal(
                block.packed_main, block.scales, rv, ri, config, d_head
            )

        if config.residual_mode == "none":
            return decode_k_metal(
                block.packed_main, block.scales, None, None, config, d_head
            )

    # Fallback to python
    def dummy_dequantize_main(packed_main, scales, *, config):
        from turboquant.core.quantizer import dequantize_groups

        d_pad = (
            (d_head + config.k_group_size - 1)
            // config.k_group_size
            * config.k_group_size
        )
        return dequantize_groups(
            packed_main, scales, config.k_bits, config.k_group_size, d_pad
        )

    return decode_k_block(block, config=config, dequantize_main=dummy_dequantize_main)
