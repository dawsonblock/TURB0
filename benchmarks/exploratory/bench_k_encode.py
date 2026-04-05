import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import time

import mlx.core as mx

from turboquant import TurboQuantConfig
from turboquant.core.pipeline import encode_k_block
from turboquant.integrations.mlx.cache_adapter import (
    dummy_dequantize_main,
    dummy_quantize_main,
)


def run_bench():
    config = TurboQuantConfig()
    x = mx.random.normal((1, 32, 128, 128))
    out = encode_k_block(x, config=config, quantize_main=dummy_quantize_main, dequantize_main=dummy_dequantize_main)
    mx.eval(out.packed_main)

    start = time.perf_counter()
    for _ in range(100):
        out = encode_k_block(x, config=config, quantize_main=dummy_quantize_main, dequantize_main=dummy_dequantize_main)
        mx.eval(out.packed_main)
    end = time.perf_counter()
    print(f"K-Encode Benchmark: {(end - start) / 100 * 1000:.2f} ms / step")

if __name__ == "__main__":
    run_bench()
