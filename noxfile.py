import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "typecheck", "tests_static"]
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ["3.9", "3.10", "3.11"]


@nox.session(python=PYTHON_VERSIONS)
def tests_static(session: nox.Session) -> None:
    """Run generic static tests without MLX dependency."""
    session.install(".[test]")
    session.run("pytest", "tests/unit_static/", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def tests_unit_mlx(session: nox.Session) -> None:
    """Run Apple Silicon MLX unit tests only (no model downloads required)."""
    session.install(".[test,apple]")
    session.run("pytest", "tests/unit_mlx/", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def tests_integration_mlx(session: nox.Session) -> None:
    """Run Apple Silicon structural integration tests (no model downloads required)."""
    session.install(".[test,apple]")
    session.run(
        "pytest",
        "tests/integration_mlx/test_path_not_dense_fallback.py",
        "tests/integration_mlx/test_cache_upgrade_roundtrip.py",
        "tests/integration_mlx/test_streaming_attention_equivalence.py",
        *session.posargs,
    )


@nox.session(python=PYTHON_VERSIONS)
def tests_mlx(session: nox.Session) -> None:
    """Run the full Apple-Silicon MLX test surface (unit + integration combined)."""
    session.install(".[test,apple]")
    session.run(
        "pytest",
        "--cov=turboquant",
        "--cov-report=term-missing",
        "tests/unit_mlx/",
        "tests/integration_mlx/",
        *session.posargs,
    )


@nox.session(python="3.11")
def lint(session: nox.Session) -> None:
    """Run linting using ruff."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.11")
def typecheck(session: nox.Session) -> None:
    """Run type checking with mypy."""
    session.install("mypy", ".")
    session.run("mypy", "turboquant/", "mlx_lm/")
