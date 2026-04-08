#!/usr/bin/env python3
"""
tools/audit_vendored_surface.py — mlx_lm integration-boundary audit.

This repo no longer vendors an in-tree ``mlx_lm/`` source tree. This script
audits the retained boundary docs instead:

  1. ``VENDORED_MLX_LM.md`` must explicitly describe continuity-only naming and
     the current upstream monkey-patching model.
  2. ``docs/vendored-upstream-boundary.md`` must remain the canonical explainer
     and must list the same patched upstream hooks as the continuity stub.
  3. The canonical integration entry point must still exist.
  4. Declared live repo paths in the boundary docs must exist and must not
     point at a nonexistent in-tree ``mlx_lm/`` checkout.

Usage
-----
    python tools/audit_vendored_surface.py           # human-readable
    python tools/audit_vendored_surface.py --json    # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTINUITY_DOC = REPO_ROOT / "VENDORED_MLX_LM.md"
BOUNDARY_DOC = REPO_ROOT / "docs" / "vendored-upstream-boundary.md"
PATCH_MODULE = REPO_ROOT / "turboquant" / "patch.py"
ENTRY_MODULE = REPO_ROOT / "turboquant" / "integrations" / "mlx" / "upgrade.py"

REQUIRED_HOOKS: tuple[str, ...] = (
    "mlx_lm.models.base.scaled_dot_product_attention",
    "mlx_lm.models.cache.make_prompt_cache",
    "mlx_lm.generate.generate_step",
)
REQUIRED_ENTRY = "turboquant.integrations.mlx.upgrade.upgrade_cache_list(...)"
REQUIRED_REPO_PATHS: tuple[str, ...] = (
    "turboquant/patch.py",
    "turboquant/integrations/mlx/upgrade.py",
)
CODE_TOKEN_RE = re.compile(r"`([^`]+)`")
ORDERED_LIST_RE = re.compile(r"^\d+\. ")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_section_tokens(text: str, section_name: str) -> list[str]:
    in_section = False
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if in_section:
                break
            in_section = line[3:].strip().lower() == section_name.lower()
            continue
        if in_section:
            body.append(line)

    tokens: list[str] = []
    for line in body:
        if not _is_list_item(line):
            continue
        if line.count("`") % 2 != 0:
            continue
        tokens.extend(CODE_TOKEN_RE.findall(line))
    return tokens


def _is_list_item(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith(("- ", "* ")):
        return True

    return bool(ORDERED_LIST_RE.match(stripped))


def _find_phrase_violations(text: str, phrases: tuple[str, ...]) -> list[str]:
    normalized = " ".join(text.lower().split())
    normalized_phrases = {
        phrase: " ".join(phrase.lower().split()) for phrase in phrases
    }
    return [
        phrase
        for phrase, normalized_phrase in normalized_phrases.items()
        if normalized_phrase not in normalized
    ]


def run_audit() -> dict[str, object]:
    missing_docs = [
        str(path.relative_to(REPO_ROOT))
        for path in (CONTINUITY_DOC, BOUNDARY_DOC)
        if not path.exists()
    ]
    if missing_docs:
        return {
            "ok": False,
            "missing_docs": missing_docs,
            "doc_violations": [],
            "documented_hooks": {},
            "missing_hooks": [],
            "mismatched_hooks": False,
            "canonical_entry": REQUIRED_ENTRY,
            "canonical_entry_exists": ENTRY_MODULE.exists(),
            "declared_repo_paths": [],
            "missing_repo_paths": [],
            "forbidden_repo_paths": [],
        }

    continuity_text = _read(CONTINUITY_DOC)
    boundary_text = _read(BOUNDARY_DOC)

    doc_violations: list[str] = []
    for phrase in _find_phrase_violations(
        continuity_text,
        (
            "filename is retained for continuity",
            "does not vendor `mlx_lm`",
            "canonical human architecture explainer",
            "turboquant.patch.apply_mlx_lm_patches()",
            "upgrade_cache_list(...)",
        ),
    ):
        doc_violations.append(f"VENDORED_MLX_LM.md missing phrase: {phrase}")

    for phrase in _find_phrase_violations(
        boundary_text,
        (
            "canonical human architecture explainer",
            "VENDORED_MLX_LM.md",
            "turboquant.patch.apply_mlx_lm_patches()",
            "upgrade_cache_list(...)",
        ),
    ):
        doc_violations.append(
            f"docs/vendored-upstream-boundary.md missing phrase: {phrase}"
        )

    continuity_hooks = _extract_section_tokens(continuity_text, "Patched upstream hooks")
    boundary_hooks = _extract_section_tokens(boundary_text, "Patched upstream hooks")
    hook_sets_match = set(continuity_hooks) == set(boundary_hooks)
    missing_hooks = sorted(
        (set(REQUIRED_HOOKS) - set(continuity_hooks))
        | (set(REQUIRED_HOOKS) - set(boundary_hooks))
    )

    declared_repo_paths = _extract_section_tokens(continuity_text, "Active repo touchpoints")
    missing_repo_paths = sorted(
        path for path in declared_repo_paths if not (REPO_ROOT / path).exists()
    )
    forbidden_repo_paths = sorted(
        path for path in declared_repo_paths if path == "mlx_lm" or path.startswith("mlx_lm/")
    )
    if sorted(declared_repo_paths) != sorted(REQUIRED_REPO_PATHS):
        doc_violations.append(
            "VENDORED_MLX_LM.md Active repo touchpoints must enumerate the current bounded repo paths."
        )

    patch_module_text = _read(PATCH_MODULE) if PATCH_MODULE.exists() else ""
    entry_module_text = _read(ENTRY_MODULE) if ENTRY_MODULE.exists() else ""
    if "def apply_mlx_lm_patches" not in patch_module_text:
        doc_violations.append("turboquant/patch.py no longer exposes apply_mlx_lm_patches().")
    if "upgrade_cache_list" not in entry_module_text:
        doc_violations.append(
            "turboquant/integrations/mlx/upgrade.py no longer defines upgrade_cache_list."
        )

    ok = not (
        missing_docs
        or doc_violations
        or missing_hooks
        or not hook_sets_match
        or missing_repo_paths
        or forbidden_repo_paths
        or not ENTRY_MODULE.exists()
    )

    return {
        "ok": ok,
        "missing_docs": missing_docs,
        "doc_violations": doc_violations,
        "documented_hooks": {
            "VENDORED_MLX_LM.md": continuity_hooks,
            "docs/vendored-upstream-boundary.md": boundary_hooks,
        },
        "missing_hooks": missing_hooks,
        "mismatched_hooks": not hook_sets_match,
        "canonical_entry": REQUIRED_ENTRY,
        "canonical_entry_exists": ENTRY_MODULE.exists(),
        "declared_repo_paths": declared_repo_paths,
        "missing_repo_paths": missing_repo_paths,
        "forbidden_repo_paths": forbidden_repo_paths,
    }


def _print_human(result: dict[str, object]) -> None:
    print("=== TurboQuant mlx_lm Integration-Boundary Audit ===")
    print(f"Status: {'OK' if result['ok'] else 'VIOLATIONS FOUND'}")
    print()

    if result["missing_docs"]:
        print("Missing docs:")
        for path in result["missing_docs"]:
            print(f"  {path}")

    hooks = result["documented_hooks"]
    if isinstance(hooks, dict):
        for doc_path, values in hooks.items():
            print(f"Patched upstream hooks in {doc_path}:")
            for hook in values:
                print(f"  {hook}")

    if result["doc_violations"]:
        print("\nDocument violations:")
        for violation in result["doc_violations"]:
            print(f"  {violation}")

    if result["missing_hooks"]:
        print("\nMissing required hooks:")
        for hook in result["missing_hooks"]:
            print(f"  {hook}")

    if result["mismatched_hooks"]:
        print("\nHook lists do not match between the two boundary docs.")

    print(f"\nCanonical entry: {result['canonical_entry']}")
    print(f"Canonical entry exists: {result['canonical_entry_exists']}")

    if result["declared_repo_paths"]:
        print("\nDeclared active repo touchpoints:")
        for path in result["declared_repo_paths"]:
            print(f"  {path}")

    if result["missing_repo_paths"]:
        print("\nMissing declared repo paths:")
        for path in result["missing_repo_paths"]:
            print(f"  {path}")

    if result["forbidden_repo_paths"]:
        print("\nForbidden in-tree mlx_lm repo paths:")
        for path in result["forbidden_repo_paths"]:
            print(f"  {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the current mlx_lm integration boundary against its live docs."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable output.",
    )
    args = parser.parse_args()

    result = run_audit()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
