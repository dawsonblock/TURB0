from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


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
        "release.yml must clear stale runtime-cert artifacts on the self-hosted runner."
    )
    assert "Require both release model secrets" in content, (
        "release.yml must fail closed unless both Llama and Gemma release-model secrets are set."
    )
    assert 'families != {"llama", "gemma"}' in content, (
        "release.yml must require both allowlisted families in the release manifest scope."
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
        "apple-runtime-cert.yml must clear stale runtime-cert artifacts on the self-hosted runner."
    )
    assert "runtime-cert-${{ github.sha }}" in content, (
        "apple-runtime-cert.yml must retain a SHA-scoped "
        "certification artifact upload."
    )
    assert "cert_manifest.json" in content and "darwin-arm64" in content, (
        "apple-runtime-cert.yml must fail closed unless the manifest "
        "is a PASS on darwin-arm64."
    )
