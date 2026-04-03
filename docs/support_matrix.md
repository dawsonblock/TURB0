# TurboQuant Support Matrix

TurboQuant is a research-grade KV-cache compression package for Apple-Silicon MLX inference. The supported runtime path is local Apple-Silicon validation. Custom Metal kernels are experimental and not part of the default supported runtime.

Please refer to [supported-surface.md](supported-surface.md) for the canonical and complete supported surface details.

## Model Architecture Matrix

| Model Architecture | Runtime Verified | Notes |
|:---|:---:|:---|
| Llama | ⬜ | Wiring present in `mlx_lm/models/llama.py`; certification incomplete |
| Gemma | ⬜ | Wiring present in `mlx_lm/models/gemma.py`; certification incomplete |
| Qwen | ⬜ | Exploratory/Vendored-only; uncertified |
| Mistral | ⬜ | Vendored-only; not certified |
| Phi | ⬜ | Vendored-only; not certified |
| All others | ⬜ | Unsupported; route through `base.py` dispatch automatically; uncertified |

## MLX Compatibility

Tested against:

- MLX >= 0.30.0

## Hardware

- Apple Silicon (M1/M2/M3/M4) — Darwin arm64 natively supported.
