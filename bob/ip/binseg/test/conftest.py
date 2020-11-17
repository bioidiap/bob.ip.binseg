#!/usr/bin/env python
# coding=utf-8

import pytest
import bob.extension


def pytest_configure(config):
    """This function is run once for pytest setup"""

    config.addinivalue_line(
        "markers",
        "skip_if_rc_var_not_set(name): this mark skips the test if a certain "
        "~/.bobrc variable is not set",
    )


    config.addinivalue_line("markers", "slow: this mark indicates slow tests")


def pytest_runtest_setup(item):
    """This function is run for every test candidate in this directory

    The test is run if this function returns ``None``.  To skip a test, call
    ``pytest.skip()``, specifying a reason.
    """

    # iterates over all markers for the item being examined, get the first
    # argument and accumulate these names
    rc_names = [
        mark.args[0]
        for mark in item.iter_markers(name="skip_if_rc_var_not_set")
    ]

    # checks all names mentioned are set in ~/.bobrc, otherwise, skip the test
    if rc_names:
        missing = [k for k in rc_names if (k not in bob.extension.rc)]
        if any(missing):
            pytest.skip(f"Test skipped because {', '.join(missing)} are **not** "
                    f"set in ~/.bobrc")


def rc_variable_set(name):
    pytest.mark.skipif(
        name not in bob.extension.rc,
        reason=f"Bob's RC variable '{name}' is not set",
    )
