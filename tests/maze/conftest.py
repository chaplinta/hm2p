"""Shared fixtures for maze tests."""

from __future__ import annotations

import pytest

from hm2p.maze.topology import build_rose_maze


@pytest.fixture
def maze():
    """Build the standard 7×5 Rosenberg maze for testing."""
    return build_rose_maze()
