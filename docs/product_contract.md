# TurboQuant Product Contract

This document defines the narrow supported surface TurboQuant can honestly claim today.

TurboQuant supports one canonical runtime path for allowlisted Llama and Gemma models via `upgrade_cache_list(...)` inside the `mlx_lm` decode flow. Direct `TurboQuantKCache(...)` construction exists only for internal/eval use and bypasses the support gate. Runtime upgrade events and persisted certification logs are separate layers. Benchmark numbers in this repo are historical or illustrative unless backed by saved certification artifacts. `block_tokens` is retained as a compatibility-only knob for future experimentation, but does not currently affect the attention dispatch path.

## 1. Supported Hardware
TurboQuant is designed exclusively for **Apple Silicon** (M1, M2, M3, M4 families). 
- Non-Apple platforms are supported for packaging, linting, and static analysis only.
- Inference via MLX is not supported or certified on non-Apple hardware.

## 2. Supported Runtime
The canonical runtime is the **local MLX runtime** on macOS.
- Deployment via remote inference servers or non-macOS environments is currently out of scope.
- There is no production deployment contract.
- There is no claim of stable behavior across every model wired into vendored `mlx_lm`.

## 3. Supported Model Families
Only model families explicitly listed in `turboquant/runtime/support.py` are in the wired allowlist.
- **Llama-family** (Llama 2, Llama 3, Llama 3.1) — artifact-backed Apple-arm64 PASS on the canonical path (`artifacts/runtime-cert/20260404_013136`)
- **Gemma-family** (Gemma, Gemma 2) — artifact-backed Apple-arm64 PASS on the canonical path (`artifacts/runtime-cert/20260404_013527`); the current batch quality guardrail remains Llama-scoped
- Other models (e.g., Qwen, Mistral, Phi, Falcon, Baichuan, Yi) may exist in the `mlx_lm` vendored directory, but that does not make them supported. If they are not in the allowlist, `upgrade_cache_list(...)` rejects them.

## 4. Canonical Import Surfaces
To ensure long-term compatibility, users must only import from:
- `turboquant.*` (Core API)
- `turboquant.integrations.mlx.*` (MLX Integration)

Root-level `integrations/` are legacy compatibility shims and will be removed in a future release.

Direct `TurboQuantKCache(...)` construction is not part of the supported
public runtime surface. It remains available only for eval, compatibility, and
test helpers; callers who want the supported path must use `upgrade_cache_list(...)`.

## 5. Secondary Surfaces And Event Split

TurboQuant has one canonical runtime path and a small set of documented
secondary surfaces. Those secondary surfaces exist, but they are not part of
the main support claim:

- The canonical runtime path upgrades dense caches through
	`upgrade_cache_list(...)` and enforces the model-family allowlist.
- Some eval and compatibility helpers still construct `TurboQuantKCache`
	directly. These bypass the support gate and are not the supported public
	entry point.
- Runtime upgrade decisions and JSONL persistence are intentionally split:
	`turboquant.integrations.mlx.upgrade.CacheUpgradeEvent` is the lightweight
	runtime result type, while `turboquant.runtime.events.EventLog` and its event
	types are the persistence-side certification surface.
- Certification or benchmark flows can explicitly convert runtime upgrade
	decisions into persistence-side events through
	`record_runtime_upgrade_events(...)` before calling
	`MetricsTracker.write(event_log=...)`.
- The canonical decode path does not auto-persist runtime upgrade events.
	Writing `events.jsonl` remains an explicit certification workflow through
	`MetricsTracker.write(event_log=...)`.

## 6. Experimental Features
- **Metal Kernels:** Custom Metal kernels (invoked via `TQ_USE_METAL=1`) are **experimental**. The default certified path uses the standard MLX Python/C++ boundary.
- **Exploratory Presets:** Any configuration not reachable via `TurboQuantConfig.from_preset()` is considered exploratory.

## 7. Runtime Certification

> **STATUS: NARROW APPLE-ARM64 CERTIFICATION ARTIFACTS EXIST.** Retained PASS manifests exist at
> `artifacts/runtime-cert/20260404_013136` for `llama` and
> `artifacts/runtime-cert/20260404_013527` for `gemma`.
> These artifacts prove the canonical `upgrade_cache_list(...)` path on Apple Silicon for the
> stages recorded in each manifest; they do not certify production readiness, unsupported families,
> or experimental Metal kernels.

"Full TurboQuant" status requires artifact-backed evidence generated via `make certify-apple-runtime`.
- Generic CI passes do not constitute runtime certification.
- Release publish must validate a PASS `cert_manifest.json` from `./scripts/certify_apple_runtime.sh`.
- The retained Llama artifact includes real-model smoke, batch quality guardrail, long-context stability, and dense/TQ benchmark sweeps.
- The retained Gemma artifact includes real-model smoke and dense/TQ benchmark sweeps; the batch quality guardrail remains Llama-scoped.
- The repo remains a narrow release candidate, not a production runtime.
