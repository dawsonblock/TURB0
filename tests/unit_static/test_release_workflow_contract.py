import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _extract_inline_python_blocks(rel_path: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_block = False

    for line in _read(rel_path).splitlines():
        stripped = line.strip()
        if stripped == "python - <<'PY'":
            in_block = True
            current = []
            continue
        if in_block and stripped == "PY":
            blocks.append(textwrap.dedent("\n".join(current)))
            in_block = False
            current = []
            continue
        if in_block:
            current.append(line)

    return blocks


def _extract_job_block(rel_path: str, job_name: str) -> str:
    lines = _read(rel_path).splitlines()
    marker = f"  {job_name}:"
    start_index: int | None = None

    for index, line in enumerate(lines):
        if line == marker:
            start_index = index + 1
            break

    assert start_index is not None, f"could not find job {job_name!r} in {rel_path}"

    block: list[str] = []
    for line in lines[start_index:]:
        if line.startswith("  ") and not line.startswith("    "):
            break
        block.append(line)

    return "\n".join(block)


def test_release_workflow_requires_single_verified_publish_path() -> None:
    content = _read(".github/workflows/release.yml")
    verify_static = _extract_job_block(
        ".github/workflows/release.yml",
        "verify-static",
    )
    publish = _extract_job_block(
        ".github/workflows/release.yml",
        "publish",
    )

    assert not (REPO_ROOT / ".github/workflows/python-publish.yml").exists(), (
        "release.yml must be the only publish workflow."
    )
    assert not (REPO_ROOT / ".github/workflows/python-app.yml").exists(), (
        "static-ci.yml plus release.yml must replace the legacy python-app workflow."
    )
    assert "certify-apple-runtime:" in content, (
        "release.yml must contain a self-hosted Apple certification job."
    )
    assert (
        "needs: [verify-static, certify-apple-runtime]" in publish
    ), (
        "release.yml publish job must depend on static checks and Apple certification."
    )
    assert "if: ${{ !contains(github.ref_name, '-rc') }}" in publish, (
        "release.yml publish job must skip RC tags so release-candidate tags "
        "exercise gates without publishing."
    )
    assert "environment: pypi" in publish, (
        "release.yml publish job must declare the pypi environment so "
        "trusted publishing emits the expected environment claim."
    )
    assert (
        "Validate certification manifest" in content and "cert_manifest.json" in content
    ), "release.yml must validate a real cert_manifest.json before publish."
    assert "Clear stale certification artifacts" in content, (
        "release.yml must clear stale runtime-cert artifacts on the self-hosted runner."
    )
    assert "Require both release model secrets" in content, (
        "release.yml must fail closed unless both Llama and Gemma "
        "release-model secrets are set."
    )
    assert "Select Apple runner Python" in content, (
        "release.yml self-hosted Apple jobs must bootstrap a system Python "
        "instead of relying on setup-python."
    )
    assert "python tools/verify_dist_contents.py" in verify_static, (
        "verify-static must validate the built wheel/sdist boundary before "
        "publishing or releasing them."
    )
    assert (
        "actions/upload-artifact@v4" in verify_static
        and "name: turboquant-dist" in verify_static
    ), "verify-static must upload the verified distributions for the publish job."
    assert (
        "actions/download-artifact@v4" in publish and "name: turboquant-dist" in publish
    ), "publish must reuse the verified dist/* artifact instead of rebuilding."
    assert "python -m build" not in publish, (
        "publish must not rebuild distributions after verification."
    )


def test_release_workflow_manifest_validation_is_contract_driven() -> None:
    blocks = _extract_inline_python_blocks(".github/workflows/release.yml")
    manifest_block = next(
        (
            block
            for block in blocks
            if "certification_scope" in block and "required_release_artifacts" in block
        ),
        "",
    )

    assert manifest_block, (
        "release.yml must include an inline manifest validator that checks "
        "scope and required artifact files."
    )
    assert '{"llama", "gemma"}' in manifest_block, (
        "release.yml manifest validation must require both allowlisted families."
    )
    assert "contract != repo_contract" in manifest_block, (
        "release.yml must compare the retained contract snapshot against "
        "the repo contract."
    )
    assert "required_release_artifacts" in manifest_block, (
        "release.yml must validate the contract-driven required release artifact set."
    )


def test_static_ci_verifies_packaging_boundary() -> None:
    content = _read(".github/workflows/static-ci.yml")
    package_job = _extract_job_block(
        ".github/workflows/static-ci.yml",
        "package-and-syntax",
    )

    assert "python tools/verify_dist_contents.py" in package_job, (
        "static-ci.yml must verify the built wheel/sdist boundary after build."
    )
    assert "turboquant-dist-py${{ matrix.python-version }}" in package_job, (
        "static-ci.yml should archive matrix builds under distinct artifact names."
    )


def test_apple_runtime_workflow_validates_manifest() -> None:
    content = _read(".github/workflows/apple-runtime-cert.yml")
    full_cert = _extract_job_block(
        ".github/workflows/apple-runtime-cert.yml",
        "full-certification",
    )
    blocks = _extract_inline_python_blocks(".github/workflows/apple-runtime-cert.yml")
    manifest_block = next(
        (
            block
            for block in blocks
            if "required_release_artifacts" in block and "certification_scope" in block
        ),
        "",
    )

    assert "run only the structural tier" in content, (
        "apple-runtime-cert.yml comments must say PRs only exercise the "
        "structural tier."
    )
    assert (
        "Full certification runs only on push to main or explicit "
        "manual dispatch." in content
    ), "apple-runtime-cert.yml comments must not imply full certification runs on PRs."
    assert "Validate certification manifest" in content, (
        "apple-runtime-cert.yml must validate the generated cert_manifest.json."
    )
    assert "Clear stale certification artifacts" in content, (
        "apple-runtime-cert.yml must clear stale runtime-cert artifacts on "
        "the self-hosted runner."
    )
    assert "runtime-cert-${{ github.sha }}" in content, (
        "apple-runtime-cert.yml must retain a SHA-scoped certification artifact upload."
    )
    assert "cert_manifest.json" in content and "darwin-arm64" in content, (
        "apple-runtime-cert.yml must fail closed unless the manifest "
        "is a PASS on darwin-arm64."
    )
    assert "Select Apple runner Python" in content, (
        "apple-runtime-cert.yml self-hosted jobs must bootstrap a system "
        "Python instead of relying on setup-python."
    )
    assert "Require both release model secrets" in full_cert, (
        "full-certification must fail closed unless both release-model secrets are set."
    )
    assert (
        "github.event_name == 'push'" in full_cert
        and "inputs.run_model_stages" in full_cert
    ), "full-certification must remain limited to push-to-main or manual dispatch."
    assert manifest_block, (
        "apple-runtime-cert.yml must include an inline manifest validator "
        "with family and artifact checks."
    )
    assert '{"llama", "gemma"}' in manifest_block, (
        "apple-runtime-cert.yml must require both allowlisted families in "
        "the certification manifest scope."
    )
    assert "required_release_artifacts" in manifest_block, (
        "apple-runtime-cert.yml must validate the contract-driven required "
        "artifact set."
    )


def test_workflow_inline_python_blocks_compile() -> None:
    for rel_path in (
        ".github/workflows/release.yml",
        ".github/workflows/apple-runtime-cert.yml",
    ):
        for block in _extract_inline_python_blocks(rel_path):
            compile(block, rel_path, "exec")
