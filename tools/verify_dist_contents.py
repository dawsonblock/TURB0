#!/usr/bin/env python3
"""
tools/verify_dist_contents.py — fail-closed audit for built wheel/sdist output.

The wheel intentionally ships the bounded ``turboquant`` package together with
the vendored ``mlx_lm`` tree, ``turboquant/contract.json``, and
``mlx_lm/py.typed``. Shipped Python runtime modules that rely on colocated
non-Python assets must ship those assets in the same built artifact. Top-level
``tests/``, ``benchmarks/``, and retained runtime evidence under ``artifacts/``
remain outside the published artifact boundary, while ``docs/*.md`` remain
source-distribution-only for human review.

Usage
-----
    python tools/verify_dist_contents.py
    python tools/verify_dist_contents.py --dist-dir dist --json

Exit codes
----------
  0  The built artifacts match the intended boundary.
  1  One or more required files are missing or forbidden paths leaked in.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST_DIR = REPO_ROOT / "dist"

REQUIRED_SHARED_MEMBERS: tuple[str, ...] = (
    "turboquant/contract.json",
    "mlx_lm/__init__.py",
    "mlx_lm/py.typed",
)
REQUIRED_WHEEL_PREFIXES: tuple[str, ...] = (
    "mlx_lm/models/",
)
REQUIRED_SDIST_ONLY_MEMBERS: tuple[str, ...] = (
    "docs/product_contract.md",
    "docs/support_matrix.md",
    "docs/supported-surface.md",
    "docs/contract_status.md",
)
REQUIRED_COLOCATED_ASSETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "turboquant/experimental/kernels/metal/runtime.py",
        ("turboquant/experimental/kernels/metal/decode_k.metal",),
    ),
)
FORBIDDEN_TOP_LEVEL_PREFIXES: tuple[str, ...] = (
    "tests/",
    "benchmarks/",
    "artifacts/",
)


def _find_single(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected exactly one "
            f"{pattern} in {dist_dir}, found {len(matches)}"
        )
    return matches[0]


def _read_wheel_members(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path) as archive:
        return tuple(archive.namelist())


def _strip_sdist_root(member: str) -> str:
    parts = member.split("/", 1)
    return parts[1] if len(parts) == 2 else member


def _read_sdist_members(path: Path) -> tuple[str, ...]:
    with tarfile.open(path, "r:gz") as archive:
        return tuple(
            _strip_sdist_root(member)
            for member in archive.getnames()
            if member
        )


def _missing_members(
    members: tuple[str, ...],
    required: tuple[str, ...],
) -> list[str]:
    member_set = set(members)
    return sorted(path for path in required if path not in member_set)


def _missing_prefixes(
    members: tuple[str, ...],
    required: tuple[str, ...],
) -> list[str]:
    return sorted(
        prefix
        for prefix in required
        if not any(member.startswith(prefix) for member in members)
    )


def _forbidden_members(members: tuple[str, ...]) -> list[str]:
    return sorted(
        member
        for member in members
        if any(
            member.startswith(prefix)
            for prefix in FORBIDDEN_TOP_LEVEL_PREFIXES
        )
    )


def _missing_colocated_assets(
    members: tuple[str, ...],
    requirements: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[str]:
    member_set = set(members)
    missing: list[str] = []
    for module_path, asset_paths in requirements:
        if module_path not in member_set:
            continue
        for asset_path in asset_paths:
            if asset_path not in member_set:
                missing.append(f"{module_path} -> {asset_path}")
    return sorted(missing)


def run_audit(dist_dir: Path = DEFAULT_DIST_DIR) -> dict[str, object]:
    wheel_path = _find_single(dist_dir, "*.whl")
    sdist_path = _find_single(dist_dir, "*.tar.gz")

    wheel_members = _read_wheel_members(wheel_path)
    sdist_members = _read_sdist_members(sdist_path)

    wheel_missing = _missing_members(wheel_members, REQUIRED_SHARED_MEMBERS)
    wheel_missing_prefixes = _missing_prefixes(
        wheel_members,
        REQUIRED_WHEEL_PREFIXES,
    )
    wheel_missing_colocated_assets = _missing_colocated_assets(
        wheel_members,
        REQUIRED_COLOCATED_ASSETS,
    )
    wheel_forbidden = _forbidden_members(wheel_members)

    sdist_missing = _missing_members(
        sdist_members,
        REQUIRED_SHARED_MEMBERS + REQUIRED_SDIST_ONLY_MEMBERS,
    )
    sdist_missing_colocated_assets = _missing_colocated_assets(
        sdist_members,
        REQUIRED_COLOCATED_ASSETS,
    )
    sdist_forbidden = _forbidden_members(sdist_members)

    ok = not any(
        (
            wheel_missing,
            wheel_missing_prefixes,
            wheel_missing_colocated_assets,
            wheel_forbidden,
            sdist_missing,
            sdist_missing_colocated_assets,
            sdist_forbidden,
        )
    )

    return {
        "ok": ok,
        "dist_dir": dist_dir.as_posix(),
        "wheel": {
            "path": wheel_path.as_posix(),
            "missing_members": wheel_missing,
            "missing_prefixes": wheel_missing_prefixes,
            "missing_colocated_assets": wheel_missing_colocated_assets,
            "forbidden_members": wheel_forbidden,
        },
        "sdist": {
            "path": sdist_path.as_posix(),
            "missing_members": sdist_missing,
            "missing_colocated_assets": sdist_missing_colocated_assets,
            "forbidden_members": sdist_forbidden,
        },
    }


def _print_human(result: dict[str, object]) -> None:
    print("=== TurboQuant Distribution Boundary Audit ===")
    print(f"Status: {'OK' if result['ok'] else 'VIOLATIONS FOUND'}")
    print(f"dist/: {result['dist_dir']}")

    for label in ("wheel", "sdist"):
        payload = result[label]
        assert isinstance(payload, dict)
        print(f"\n{label.upper()}: {payload['path']}")
        missing_members = payload.get("missing_members", [])
        if missing_members:
            print("  Missing members:")
            for member in missing_members:
                print(f"    {member}")
        missing_prefixes = payload.get("missing_prefixes", [])
        if missing_prefixes:
            print("  Missing prefixes:")
            for prefix in missing_prefixes:
                print(f"    {prefix}")
        missing_colocated_assets = payload.get("missing_colocated_assets", [])
        if missing_colocated_assets:
            print("  Missing colocated assets:")
            for dependency in missing_colocated_assets:
                print(f"    {dependency}")
        forbidden_members = payload.get("forbidden_members", [])
        if forbidden_members:
            print("  Forbidden members:")
            for member in forbidden_members:
                print(f"    {member}")

    if result["ok"]:
        print(
            "\nBuilt artifacts match the intended TurboQuant shipping "
            "boundary."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate built wheel/sdist contents against the intended "
            "boundary."
        )
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_DIST_DIR,
        help=(
            "Directory containing exactly one wheel and one source "
            "distribution."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable output.",
    )
    args = parser.parse_args()

    result = run_audit(args.dist_dir)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
