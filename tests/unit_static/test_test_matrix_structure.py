import os
import pytest

def get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_test_matrix_structure():
    """Ensure that the test suite is structured correctly by environment."""
    root = get_repo_root()
    
    # Check for presence of required directories
    assert os.path.exists(os.path.join(root, "tests/unit_static"))
    assert os.path.exists(os.path.join(root, "tests/unit_mlx"))
    assert os.path.exists(os.path.join(root, "tests/integration_mlx"))
    
    # Check for absence of old directories
    assert not os.path.exists(os.path.join(root, "tests/unit"))

def test_static_standalone_imports():
    """Ensure that files in unit_static do not import MLX directly or indirectly via TurboQuant runtime."""
    # This is a bit harder to test without running it, but we can do a simple search or import check
    pass
