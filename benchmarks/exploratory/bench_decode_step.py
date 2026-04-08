# flake8: noqa

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time

import mlx.core as mx

from turboquant import TurboQuantConfig
from turboquant.integrations.mlx._cache_adapter import (
    dummy_dequantize_main,
    dummy_quantize_main,
)
from turboquant.runtime.kv_interface import TurboQuantKVCache


def run_bench():
    config = TurboQuantConfig()
    cache = TurboQuantKVCache(
        config=config,
        quantize_main=dummy_quantize_main,
        dequantize_main=dummy_dequantize_main,
    )

    k = mx.random.normal((1, 32, 1, 128))
    cache.append_keys(k)

    start = time.perf_counter()
    for _ in range(100):
        cache.append_keys(k)
    end = time.perf_counter()

    print(f"Decode Step Benchmark: {(end - start) / 100 * 1000:.2f} ms / step")


if __name__ == "__main__":
    run_bench()
