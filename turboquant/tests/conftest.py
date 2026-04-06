"""
Compatibility stub.  Canonical tests now live under ``tests/``.

The maintained test layout is:
- ``tests/unit_static/`` for portable structural governance checks
- ``tests/unit_mlx/`` for Apple-Silicon MLX unit coverage
- ``tests/integration_mlx/`` for Apple-Silicon structural and runtime tests

This conftest is retained so that ``pytest turboquant/tests/`` still adds the
project root to ``sys.path`` when someone runs from that directory directly.
Canonical test command: ``pytest tests/``
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
