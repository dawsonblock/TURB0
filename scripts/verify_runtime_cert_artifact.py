#!/usr/bin/env python3
"""Validate a retained runtime-cert artifact against the repo contract.

This helper is intended for downloaded GitHub Actions artifacts as well as
locally retained runtime-cert directories. It fails unless the artifact proves
the final Apple-Silicon release boundary the repo claims: PASS on darwin-arm64,
both llama and gemma in scope, a retained contract snapshot matching the repo
contract, and all contract-driven required release artifacts present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, cast
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "turboquant" / "contract.json"
REQUIRED_FAMILIES = {"llama", "gemma"}
JsonDict = dict[str, Any]


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return cast(JsonDict, payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_artifact_root(base: Path) -> Path:
    manifests = sorted(base.rglob("cert_manifest.json"))
    if not manifests:
        raise SystemExit(
            f"No cert_manifest.json found under artifact path: {base}"
        )

    roots = sorted({path.parent for path in manifests})
    if len(roots) != 1:
        formatted = ", ".join(str(path) for path in roots)
        raise SystemExit(
            "Expected exactly one runtime-cert root containing cert_manifest.json, "
            f"found {len(roots)}: {formatted}"
        )
    return roots[0]


def _materialize_artifact(path: Path) -> tuple[Path, Optional[TemporaryDirectory]]:
    if path.is_dir():
        return _find_artifact_root(path), None

    if path.is_file() and path.suffix == ".zip":
        tempdir = TemporaryDirectory()
        with ZipFile(path) as archive:
            archive.extractall(tempdir.name)
        return _find_artifact_root(Path(tempdir.name)), tempdir

    raise SystemExit(
        "Artifact path must be either a runtime-cert directory or a .zip file: "
        f"{path}"
    )


def _validate_artifact(artifact_root: Path) -> list[str]:
    errors: list[str] = []
    repo_contract = _load_json(CONTRACT_PATH)
    manifest_path = artifact_root / "cert_manifest.json"
    contract_path = artifact_root / "contract.json"

    if not manifest_path.is_file():
        return [f"Missing cert_manifest.json in artifact root: {artifact_root}"]
    if not contract_path.is_file():
        return [f"Missing contract.json in artifact root: {artifact_root}"]

    manifest = _load_json(manifest_path)
    artifact_contract = _load_json(contract_path)

    if manifest.get("result") != "PASS":
        errors.append(
            f"cert_manifest.json result must be PASS, got {manifest.get('result')!r}"
        )
    if manifest.get("platform") != "darwin-arm64":
        errors.append(
            "cert_manifest.json platform must be darwin-arm64, got "
            f"{manifest.get('platform')!r}"
        )

    families = set(manifest.get("certification_scope", {}).get("families", []))
    if families != REQUIRED_FAMILIES:
        errors.append(
            "cert_manifest.json families must be exactly ['gemma', 'llama'], got "
            f"{sorted(families)!r}"
        )

    if artifact_contract != repo_contract:
        errors.append(
            f"artifact contract {contract_path} does not match repo contract {CONTRACT_PATH}"
        )

    required_artifacts = repo_contract["canonical_runtime"]["required_release_artifacts"]
    missing = [
        name for name in required_artifacts if not (artifact_root / name).exists()
    ]
    if missing:
        errors.append(
            "artifact is missing required release files: " + ", ".join(missing)
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a runtime-cert artifact directory or zip."
    )
    parser.add_argument(
        "artifact",
        help="Path to a runtime-cert artifact directory or downloaded zip file.",
    )
    args = parser.parse_args()

    artifact_path = Path(args.artifact).expanduser().resolve()
    artifact_root: Path
    extracted: Optional[TemporaryDirectory]
    artifact_root, extracted = _materialize_artifact(artifact_path)
    try:
        errors = _validate_artifact(artifact_root)
        if errors:
            print("Runtime certification artifact validation failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1

        manifest_path = artifact_root / "cert_manifest.json"
        manifest = _load_json(manifest_path)
        required_count = len(
            _load_json(CONTRACT_PATH)["canonical_runtime"]["required_release_artifacts"]
        )
        families = sorted(manifest["certification_scope"]["families"])

        print(f"Validated runtime certification artifact: {artifact_root}")
        print(f"Manifest SHA-256: {_sha256(manifest_path)}")
        print(f"Result: {manifest['result']}")
        print(f"Platform: {manifest['platform']}")
        print(f"Families: {', '.join(families)}")
        print(f"Required release artifacts present: {required_count}")
        print("Contract snapshot matches repo contract: yes")
        return 0
    finally:
        if extracted is not None:
            extracted.cleanup()


if __name__ == "__main__":
    sys.exit(main())