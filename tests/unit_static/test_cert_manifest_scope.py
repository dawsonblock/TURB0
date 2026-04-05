import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WRITE_CERT_MANIFEST = REPO_ROOT / "scripts" / "write_cert_manifest.py"


def test_write_cert_manifest_records_family_scope(tmp_path) -> None:
    artifact_dir = tmp_path / "runtime-cert"
    artifact_dir.mkdir()
    (artifact_dir / "preflight.json").write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(WRITE_CERT_MANIFEST),
            "--artifact-dir",
            str(artifact_dir),
            "--passed",
            "7",
            "--failed",
            "0",
            "--skipped",
            "0",
            "--unimplemented",
            "0",
            "--out-of-scope",
            "3",
            "--total",
            "7",
            "--family",
            "llama",
            "--turboquant-version",
            "0.2.2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    manifest = json.loads(
        (artifact_dir / "cert_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "3"
    assert manifest["result"] == "PASS"
    assert manifest["certification_scope"]["families"] == ["llama"]
    assert manifest["stages"]["out_of_scope"] == 3


def test_write_cert_manifest_requires_real_model_scope_for_pass(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "runtime-cert"
    artifact_dir.mkdir()
    (artifact_dir / "preflight.json").write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(WRITE_CERT_MANIFEST),
            "--artifact-dir",
            str(artifact_dir),
            "--passed",
            "7",
            "--failed",
            "0",
            "--skipped",
            "0",
            "--unimplemented",
            "0",
            "--out-of-scope",
            "4",
            "--total",
            "7",
            "--turboquant-version",
            "0.2.2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stderr or result.stdout

    manifest = json.loads(
        (artifact_dir / "cert_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["result"] == "FAIL"
    assert manifest["certification_scope"]["families"] == []


def test_certify_script_marks_unselected_families_out_of_scope() -> None:
    content = (REPO_ROOT / "scripts" / "certify_apple_runtime.sh").read_text(
        encoding="utf-8"
    )

    assert "OUT OF SCOPE" in content, (
        "certify_apple_runtime.sh must mark unselected families "
        "as out of scope."
    )
    assert "No real-model certification scope selected" in content, (
        "certify_apple_runtime.sh must fail closed when no real-model "
        "family is selected."
    )
