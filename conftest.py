"""Shared pytest configuration."""

from __future__ import annotations


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.slow tests unless --runslow is passed."""
    import pytest

    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (e.g. full PyMC sampling)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect by default)")
