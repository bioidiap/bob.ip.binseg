#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib

import pytest


@pytest.fixture
def datadir(request) -> pathlib.Path:
    """Returns the directory in which the test is sitting."""
    return pathlib.Path(request.module.__file__).parents[0] / "data"


def pytest_configure(config):
    """This function is run once for pytest setup."""
    config.addinivalue_line(
        "markers",
        "skip_if_rc_var_not_set(name): this mark skips the test if a certain "
        "~/.config/deepdraw.toml variable is not set",
    )

    config.addinivalue_line("markers", "slow: this mark indicates slow tests")


def pytest_runtest_setup(item):
    """This function is run for every test candidate in this directory.

    The test is run if this function returns ``None``.  To skip a test,
    call ``pytest.skip()``, specifying a reason.
    """
    from deepdraw.common.utils.rc import load_rc

    rc = load_rc()

    # iterates over all markers for the item being examined, get the first
    # argument and accumulate these names
    rc_names = [
        mark.args[0]
        for mark in item.iter_markers(name="skip_if_rc_var_not_set")
    ]

    # checks all names mentioned are set in ~/.config/deepdraw.toml, otherwise,
    # skip the test
    if rc_names:
        missing = [k for k in rc_names if rc.get(k) is None]
        if any(missing):
            pytest.skip(
                f"Test skipped because {', '.join(missing)} is **not** "
                f"set in ~/.config/deepdraw.toml"
            )


def rc_variable_set(name):
    from deepdraw.common.utils.rc import load_rc

    rc = load_rc()
    pytest.mark.skipif(
        name not in rc,
        reason=f"RC variable '{name}' is not set",
    )


@pytest.fixture(scope="session")
def temporary_basedir(tmp_path_factory):
    return tmp_path_factory.mktemp("test-cli")


@pytest.fixture(scope="session")
def stare_datadir(tmp_path_factory) -> pathlib.Path:
    from deepdraw.common.utils.rc import load_rc

    database_dir = load_rc().get("datadir.stare")
    if database_dir is not None:
        return pathlib.Path(database_dir)

    # else, we must extract the LFS component
    archive = (
        pathlib.Path(__file__).parents[0]
        / "data"
        / "deepdraw-ci-assets"
        / "test-database.zip"
    )
    assert archive.exists(), (
        f"Neither datadir.stare is set on the global configuration, "
        f"(typically ~/.config/deepdraw.toml), or it is possible to detect "
        f"the presence of {archive}' (did you git submodule init --update "
        f"this submodule?)"
    )

    database_dir = tmp_path_factory.mktemp("stare_datadir")

    import zipfile

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(database_dir)

    return database_dir
