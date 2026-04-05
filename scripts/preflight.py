from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "turboquant" / "contract.json"


def load_contract() -> dict[str, Any]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract: dict[str, Any] = cast(dict[str, Any], payload)
    return contract


def _parse_version(version: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for raw_part in version.split(".")[:3]:
        digits = ""
        for ch in raw_part:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    parsed: tuple[int, int, int] = (parts[0], parts[1], parts[2])
    return parsed


def check_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect_mlx_version() -> str | None:
    try:
        import mlx
    except ImportError:
        return None

    version = getattr(mlx, "__version__", None)
    if isinstance(version, str) and version:
        return version

    try:
        metadata_version = importlib.metadata.version("mlx")
        return cast(str, metadata_version)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"


def check_mlx_version() -> str | None:
    """Backward-compatible alias for the MLX version detector."""
    return detect_mlx_version()


def _python_bounds(
    contract: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int]]:
    python_contract = contract["canonical_runtime"]["python"]
    minimum = tuple(int(part) for part in python_contract["min"].split("."))
    maximum = tuple(int(part) for part in python_contract["max"].split("."))
    return (minimum[0], minimum[1]), (maximum[0], maximum[1])


def is_supported_python_version(version_info: tuple[int, int]) -> bool:
    minimum, maximum = _python_bounds(load_contract())
    return minimum <= version_info <= maximum


def is_supported_mlx_version(version: str | None) -> bool | None:
    if version is None:
        return None
    if version == "unknown":
        return False

    contract = load_contract()
    mlx_contract = contract["canonical_runtime"]["mlx"]
    minimum = _parse_version(mlx_contract["min"])
    maximum = _parse_version(mlx_contract["max_exclusive"])
    current = _parse_version(version)
    return minimum <= current < maximum


def _supported_python_label(contract: dict[str, Any]) -> str:
    python_contract = contract["canonical_runtime"]["python"]
    return f"{python_contract['min']}-{python_contract['max']}"


def _supported_mlx_label(contract: dict[str, Any]) -> str:
    mlx_contract = contract["canonical_runtime"]["mlx"]
    return f">={mlx_contract['min']},<{mlx_contract['max_exclusive']}"


def _import_turboquant() -> tuple[str | None, bool, list[str]]:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    spec = importlib.util.find_spec("turboquant")
    if spec is None:
        return None, False, ["Cannot import turboquant"]

    try:
        import turboquant

        return getattr(turboquant, "__version__", "dev"), True, []
    except Exception as exc:
        return None, False, [f"Import failed: {exc}"]


def collect_results() -> dict[str, Any]:
    contract = load_contract()
    mlx_version = detect_mlx_version()
    turboquant_version, importable, errors = _import_turboquant()

    return {
        "apple_silicon": check_apple_silicon(),
        "python_version": sys.version.split("\n")[0],
        "python_supported": is_supported_python_version(sys.version_info[:2]),
        "supported_python": _supported_python_label(contract),
        "mlx_version": mlx_version,
        "mlx_version_supported": is_supported_mlx_version(mlx_version),
        "supported_mlx": _supported_mlx_label(contract),
        "turboquant_version": turboquant_version,
        "turboquant_importable": importable,
        "platform": platform.platform(),
        "errors": errors,
    }


def strict_failures(results: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    if not results["apple_silicon"]:
        failures.append(
            "TurboQuant runtime certification requires Apple Silicon "
            "(darwin-arm64)."
        )
    if not results["python_supported"]:
        failures.append(
            "Python version is outside the supported range "
            f"{results['supported_python']}."
        )

    mlx_version = results["mlx_version"]
    if mlx_version is None:
        failures.append("MLX is not installed.")
    elif not results["mlx_version_supported"]:
        failures.append(
            "MLX version is outside the supported range "
            f"{results['supported_mlx']} (found {mlx_version})."
        )

    if not results["turboquant_importable"]:
        errors = results.get("errors", [])
        if errors:
            failures.extend(str(error) for error in errors)
        else:
            failures.append("TurboQuant import failed.")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TurboQuant environment preflight"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail closed unless the Apple-Silicon runtime contract is "
            "satisfied"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output",
    )
    args = parser.parse_args()

    results = collect_results()
    failures = strict_failures(results) if args.strict else []
    results["strict_ready"] = not failures
    results["strict_failures"] = failures

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for key, value in results.items():
            print(f"{key}: {value}")
        for failure in failures:
            print(f"ERROR: {failure}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
