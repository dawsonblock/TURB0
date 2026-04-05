"""Minimal self-contained transformer model for TurboQuant integration tests.

Provides a tiny model (128-dim, 2-layer, 2-head) and a character-level
tokenizer that work end-to-end with:

  - mlx_lm.models.cache.make_prompt_cache   (via model.layers)
  - mlx_lm.generate.generate_step           (returns [B,T,vocab] logits)
  - turboquant.integrations.mlx.upgrade.upgrade_cache_list
  - turboquant.runtime.attention.turboquant_streaming_attention

No weights are downloaded; parameters are random MLX arrays.
Dimensions (64-wide heads, group_size-32/64 compatible, Hadamard-safe):

  VOCAB_SIZE = 64
  N_HEADS    = 2
  HEAD_DIM   = 64   (power-of-2 for Hadamard; multiple of 32 for group_size)
  D_MODEL    = 128
  N_LAYERS   = 2
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from turboquant.runtime.attention import turboquant_streaming_attention
from turboquant.runtime.kv_interface import TurboQuantKeysView

# ---------------------------------------------------------------------------
# Hyper-parameters (do NOT change — tests may depend on these values)
# ---------------------------------------------------------------------------
VOCAB_SIZE: int = 64
N_HEADS: int = 2
HEAD_DIM: int = 64
D_MODEL: int = N_HEADS * HEAD_DIM  # 128
N_LAYERS: int = 2


# ---------------------------------------------------------------------------
# Model layers
# ---------------------------------------------------------------------------

class _TinyAttn(nn.Module):
    """Minimal multi-head attention with TurboQuant dispatch."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.scale: float = HEAD_DIM ** -0.5

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        B, T, _ = x.shape
        H, D = N_HEADS, HEAD_DIM

        q = self.q_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if isinstance(k, TurboQuantKeysView):
            # TQ path: turboquant_streaming_attention fetches v from the cache
            # internally (paper_kv mode), so we do not pass v here.
            out = turboquant_streaming_attention(q, k, scale=self.scale)
        else:
            # Dense path: standard scaled dot-product attention.
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            out = mx.softmax(scores, axis=-1) @ v

        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL))


class _TinyBlock(nn.Module):
    """Pre-norm transformer block (attention + single linear FFN)."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = _TinyAttn()
        self.norm1 = nn.RMSNorm(D_MODEL)
        self.ff = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.norm2 = nn.RMSNorm(D_MODEL)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        x = x + self.attn(self.norm1(x), cache=cache)
        x = x + self.ff(self.norm2(x))
        return x


class TinyModel(nn.Module):
    """
    A tiny (128-dim, 2-layer) MLX language model for integration tests.

    Compatible with ``make_prompt_cache`` (uses ``model.layers`` length) and
    ``generate_step`` (accepts ``(input_ids, cache=...)`` and returns
    ``[B, T, vocab_size]`` logits).

    The attention module dispatches to ``turboquant_streaming_attention`` when
    ``update_and_fetch`` returns a :class:`TurboQuantKeysView`, so the full
    TurboQuant path is exercised without a real language model.

    ``model_type`` is set to ``"llama"`` so that :func:`~mlx_lm.generate._infer_model_family`
    can resolve a supported family without requiring a real model download.
    """

    model_type: str = "llama"

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.layers = [_TinyBlock() for _ in range(N_LAYERS)]
        self.norm = nn.RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        """Forward pass.

        Args:
            input_ids: ``[B, T]`` integer token ids in ``[0, VOCAB_SIZE)``.
            cache: Per-layer cache list (``KVCache`` or ``TurboQuantKCache``),
                or ``None`` for cache-free forward.

        Returns:
            Logits ``[B, T, VOCAB_SIZE]``.
        """
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            x = layer(x, cache=c)
        return self.lm_head(self.norm(x))


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TinyTokenizer:
    """
    Character-level tokenizer mapping each character to ``[0, VOCAB_SIZE)``.

    Uses ``ord(c) % VOCAB_SIZE`` so every ASCII character maps to a valid
    token id without out-of-vocabulary errors.  Suitable for any short
    English text used in smoke tests.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) % VOCAB_SIZE for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr((i % 95) + 32) for i in ids)
