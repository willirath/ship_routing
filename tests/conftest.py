"""Pytest configuration and shared fixtures."""
from pathlib import Path

# Project root is two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
