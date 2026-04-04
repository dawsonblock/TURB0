from pathlib import Path
import textwrap


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


def test_release_workflow_requires_apple_cert_and_vendored_audit() -> None:
    content = _read(".github/workflows/release.yml")

    assert "verify-vendored-surface:" in content, (
        "release.yml must contain a dedicated vendored-surface "
        "verification job."
    )
    assert "certify-apple-runtime:" in content, (
        "release.yml must contain a self-hosted Apple certification job."
    )
    assert (
        "needs: [verify-static, verify-vendored-surface, "
        "certify-apple-runtime]" in content
    ), (
        "release.yml publish job must depend on static checks, "
        "vendored audit, and Apple certification."
    )
    assert "if: ${{ !contains(github.ref_name, '-rc') }}" in content, (
        "release.yml publish job must skip RC tags so release-candidate tags "
        "exercise gates without publishing."
    )
    assert "environment: pypi" in content, (
        "release.yml publish job must declare the pypi environment so "
        "trusted publishing emits the expected environment claim."
    )
    assert "python tools/audit_vendored_surface.py" in content, (
        "release.yml must run the vendored mlx_lm surface audit "
        "before publish."
    )
    assert (
        "Validate certification manifest" in content
        and "cert_manifest.json" in content
    ), (
        "release.yml must validate a real cert_manifest.json before publish."
    )
    assert "Clear stale certification artifacts" in content, (
        "release.yml must clear stale runtime-cert artifacts on the "
        "self-hosted runner."
    )
    assert "Require both release model secrets" in content, (
        "release.yml must fail closed unless both Llama and Gemma "
        "release-model secrets are set."
    )
    assert 'families != {"llama", "gemma"}' in content, (
        "release.yml must require both allowlisted families in the "
        "release manifest scope."
    )
    assert "Select Apple runner Python" in content, (
        "release.yml self-hosted Apple jobs must bootstrap a system Python "
        "instead of relying on setup-python."
    )


def test_static_ci_runs_vendored_surface_audit() -> None:
    content = _read(".github/workflows/static-ci.yml")

    assert "vendored-surface-audit:" in content, (
        "static-ci.yml must contain a dedicated vendored surface audit job."
    )
    assert "python tools/audit_vendored_surface.py" in content, (
        "static-ci.yml must fail on undocumented TurboQuant markers "
        "inside mlx_lm/."
    )


def test_apple_runtime_workflow_validates_manifest() -> None:
    content = _read(".github/workflows/apple-runtime-cert.yml")

    assert "Validate certification manifest" in content, (
        "apple-runtime-cert.yml must validate the generated "
        "cert_manifest.json."
    )
    assert "Clear stale certification artifacts" in content, (
        "apple-runtime-cert.yml must clear stale runtime-cert artifacts on "
        "the self-hosted runner."
    )
    assert "runtime-cert-${{ github.sha }}" in content, (
        "apple-runtime-cert.yml must retain a SHA-scoped "
        "certification artifact upload."
    )
    assert "cert_manifest.json" in content and "darwin-arm64" in content, (
        "apple-runtime-cert.yml must fail closed unless the manifest "
        "is a PASS on darwin-arm64."
    )
    assert "Select Apple runner Python" in content, (
        "apple-runtime-cert.yml self-hosted jobs must bootstrap a system "
        "Python instead of relying on setup-python."
    )


def test_workflow_inline_python_blocks_compile() -> None:
    for rel_path in (
        ".github/workflows/release.yml",
        ".github/workflows/apple-runtime-cert.yml",
    ):
        for block in _extract_inline_python_blocks(rel_path):
            compile(block, rel_path, "exec")
