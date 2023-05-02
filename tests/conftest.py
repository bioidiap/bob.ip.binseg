# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import contextlib
import io
import os
import pathlib
import tempfile
import warnings
import zipfile

import pytest
import tomli_w

from click.testing import CliRunner

"""

In your tests:

  def test_foo(cli_runner):
    r = cli_runner.invoke(mycli, ["mycommand"])
    assert r.exit_code == 0

In `some_command()`, add:

  @cli.command()
  def mycommand():
    import pytest; pytest.set_trace()

Then run via:

  $ pytest -sv --pdb-trace ...

Note any tests checking CliRunner stdout/stderr values will fail when
--pdb-trace is set.

"""


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--pdb-trace",
        action="store_true",
        default=False,
        help="Allow calling pytest.set_trace() in Click's CliRunner",
    )


class MyCliRunner(CliRunner):
    def __init__(self, *args, in_pdb=False, **kwargs) -> None:
        self._in_pdb = in_pdb
        super().__init__(*args, **kwargs)

    def invoke(self, cli, args=None, **kwargs):
        params = kwargs.copy()
        if self._in_pdb:
            params["catch_exceptions"] = False

        return super().invoke(cli, args=args, **params)

    def isolation(self, input=None, env=None, color=False):
        if self._in_pdb:
            if input or env or color:
                warnings.warn(
                    "CliRunner PDB un-isolation doesn't work if input/env/color are passed"
                )
            else:
                return self.isolation_pdb()

        return super().isolation(input=input, env=env, color=color)

    @contextlib.contextmanager
    def isolation_pdb(self):
        s = io.BytesIO(b"{stdout not captured because --pdb-trace}")
        yield (s, not self.mix_stderr and s)


@pytest.fixture
def cli_runner(request) -> MyCliRunner:
    """A wrapper round Click's test CliRunner to improve usefulness."""
    return MyCliRunner(
        # workaround Click's environment isolation so debugging works.
        in_pdb=request.config.getoption("--pdb-trace")
    )


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
    from deepdraw.utils.rc import load_rc

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
    from deepdraw.utils.rc import load_rc

    rc = load_rc()
    pytest.mark.skipif(
        name not in rc,
        reason=f"RC variable '{name}' is not set",
    )


@pytest.fixture(scope="session")
def temporary_basedir(tmp_path_factory):
    return tmp_path_factory.mktemp("test-cli")


def pytest_sessionstart(session: pytest.Session) -> None:
    """Presets the session start to ensure the STARE dataset is always
    available."""

    from deepdraw.utils.rc import load_rc

    rc = load_rc()

    database_dir = rc.get("datadir.stare")
    if database_dir is not None:
        # if the user downloaded it, use that copy
        return

    # else, we must extract the LFS component (we are likely on the CI)
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

    stare_tempdir = tempfile.TemporaryDirectory()
    rc.setdefault("datadir.stare", stare_tempdir.name)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(stare_tempdir.name)

    config_filename = "deepdraw.toml"
    with open(os.path.join(stare_tempdir.name, config_filename), "wb") as f:
        tomli_w.dump(rc.data, f)
        f.flush()

    os.environ["XDG_CONFIG_HOME"] = stare_tempdir.name

    # stash the newly created temporary directory so we can erase it when the
    key = pytest.StashKey[tempfile.TemporaryDirectory]()
    session.stash[key] = stare_tempdir
