from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_preflight_module():
    script_path = REPO_ROOT / "scripts" / "preflight.py"
    spec = importlib.util.spec_from_file_location(
        "turboquant_preflight",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_helpers_enforce_contract_bounds() -> None:
    module = _load_preflight_module()

    assert module.is_supported_python_version((3, 9))
    assert module.is_supported_python_version((3, 10))
    assert module.is_supported_python_version((3, 11))
    assert not module.is_supported_python_version((3, 8))
    assert not module.is_supported_python_version((3, 12))

    assert module.is_supported_mlx_version("0.30.0") is True
    assert module.is_supported_mlx_version("0.30.1") is True
    assert module.is_supported_mlx_version("0.29.9") is False
    assert module.is_supported_mlx_version("1.0.0") is False
    assert module.is_supported_mlx_version(None) is None


def test_preflight_collect_results_exposes_contract_state() -> None:
    module = _load_preflight_module()
    results = module.collect_results()

    assert "apple_silicon" in results
    assert "python_supported" in results
    assert "supported_python" in results
    assert "mlx_version" in results
    assert "mlx_version_supported" in results
    assert "supported_mlx" in results
    assert "turboquant_importable" in results
    assert results["supported_python"] == "3.9-3.11"
    assert results["supported_mlx"] == ">=0.30.0,<1.0.0"
    assert isinstance(results["errors"], list)


def test_preflight_strict_failures_fail_closed() -> None:
    module = _load_preflight_module()
    failures = module.strict_failures(
        {
            "apple_silicon": False,
            "python_supported": False,
            "supported_python": "3.9-3.11",
            "mlx_version": None,
            "mlx_version_supported": None,
            "supported_mlx": ">=0.30.0,<1.0.0",
            "turboquant_importable": False,
            "errors": ["Import failed: boom"],
        }
    )

    assert any("Apple Silicon" in failure for failure in failures)
    assert any(
        "Python version is outside the supported range" in failure
        for failure in failures
    )
    assert any("MLX is not installed" in failure for failure in failures)
    assert any("Import failed: boom" in failure for failure in failures)
