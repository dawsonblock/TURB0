from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _extract_toml_body(header: str) -> str:
    pattern = re.compile(
        rf"^\[{re.escape(header)}\]\n(?P<body>(?:.*\n)*?)(?=^\[|\Z)",
        re.MULTILINE,
    )
    match = pattern.search(_read("pyproject.toml"))
    assert match is not None, f"could not find [{header}] in pyproject.toml"
    return match.group("body")


def _extract_toml_list(header: str, key: str) -> tuple[str, ...]:
    body = _extract_toml_body(header)
    pattern = re.compile(
        rf"^{re.escape(key)}\s*=\s*(\[[^\n]+\])$",
        re.MULTILINE,
    )
    match = pattern.search(body)
    assert match is not None, f"could not find {key!r} in [{header}]"
    return tuple(ast.literal_eval(match.group(1)))


def _load_dist_verifier():
    script_path = REPO_ROOT / "tools" / "verify_dist_contents.py"
    spec = importlib.util.spec_from_file_location(
        "verify_dist_contents",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pyproject_intentionally_ships_vendored_boundary() -> None:
    includes = set(
        _extract_toml_list("tool.setuptools.packages.find", "include")
    )
    excludes = set(
        _extract_toml_list("tool.setuptools.packages.find", "exclude")
    )
    mlx_data = set(
        _extract_toml_list("tool.setuptools.package-data", "mlx_lm")
    )
    turboquant_data = set(
        _extract_toml_list("tool.setuptools.package-data", "turboquant")
    )

    assert includes == {"turboquant*", "mlx_lm*"}
    assert {"tests*", "benchmarks*"} <= excludes
    assert mlx_data == {"py.typed"}
    assert turboquant_data == {"contract.json"}


def test_manifest_explicitly_prunes_non_shipped_top_level_dirs() -> None:
    content = _read("MANIFEST.in")

    assert "include turboquant/contract.json" in content
    assert "recursive-include mlx_lm py.typed" in content
    assert "recursive-include docs *.md" in content
    assert "prune tests" in content
    assert "prune benchmarks" in content


def test_dist_verifier_tracks_the_same_boundary() -> None:
    verifier = _load_dist_verifier()

    assert set(verifier.REQUIRED_SHARED_MEMBERS) == {
        "turboquant/contract.json",
        "mlx_lm/__init__.py",
        "mlx_lm/py.typed",
    }
    assert set(verifier.REQUIRED_WHEEL_PREFIXES) == {"mlx_lm/models/"}
    assert set(verifier.REQUIRED_SDIST_ONLY_MEMBERS) >= {
        "docs/product_contract.md",
        "docs/support_matrix.md",
        "docs/supported-surface.md",
        "docs/contract_status.md",
    }
    assert set(verifier.FORBIDDEN_TOP_LEVEL_PREFIXES) == {
        "tests/",
        "benchmarks/",
    }
