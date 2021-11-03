# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-large", action="store_true", default=False,
        help="run large tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "large: mark test as large to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-large"):
        # --run-large given in cli: do not skip large tests
        return
    skip_large = pytest.mark.skip(reason="need --run-large option to run")
    for item in items:
        if "large" in item.keywords:
            item.add_marker(skip_large)
