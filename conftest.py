"""
conftest.py
-----------
Pytest configuration: ensures the project root and src/ are on sys.path
so tests can be run from any working directory with just `pytest tests/ -v`.
"""
import sys
import os

# Add project root and src to path
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)
