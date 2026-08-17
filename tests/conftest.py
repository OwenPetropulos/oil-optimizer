"""
conftest.py — pytest configuration for the test suite.

This file does one job: make the `oil_optimizer` package (which lives under
src/) importable from the tests. pytest runs this automatically before
collecting tests. You don't call anything here yourself.
"""
import sys
from pathlib import Path

# Add the src/ directory to Python's import path so `import oil_optimizer`
# works from inside the tests, regardless of where pytest is launched.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
