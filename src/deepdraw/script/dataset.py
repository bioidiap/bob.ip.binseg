# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import importlib.resources
import os

import click

from clapper.click import AliasedGroup, verbosity_option
from clapper.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")


def _get_supported_datasets():
    """Returns a list of supported dataset names."""
    basedir = importlib.resources.files(__name__.split(".", 1)[0]).joinpath(
        "data/"
    )

    retval = []
    for candidate in basedir.iterdir():
        if candidate.is_dir() and "__init__.py" in os.listdir(str(candidate)):
            retval.append(candidate.name)

    return set(retval)


def _get_installed_datasets() -> dict[str, str]:
    """Returns a list of installed datasets as regular expressions.

    * group(0): the name of the key for the dataset directory
    * group("name"): the short name for the dataset
    """
    from deepdraw.utils.rc import load_rc

    return dict(load_rc().get("datadir", {}))


@click.group(cls=AliasedGroup)
def dataset() -> None:
    """Commands for listing and verifying datasets."""
    pass


@dataset.command(
    epilog="""Examples:

\b
  1. To install a dataset, set up its data directory ("datadir").  For
     example, to setup access to Montgomery files you downloaded locally at
     the directory "/path/to/montgomery/files", edit the RC file (typically
     ``$HOME/.config/deepdraw.toml``), and add a line like the following:

     .. code:: toml

        [datadir]
        montgomery = "/path/to/montgomery/files"

     .. note::

        This setting **is** case-sensitive.

\b
  2. List all raw datasets supported (and configured):

     .. code:: sh

        $ deepdraw dataset list
""",
)
@verbosity_option(logger=logger, expose_value=False)
def list():
    """Lists all supported and configured datasets."""
    supported = _get_supported_datasets()
    installed = _get_installed_datasets()

    click.echo("Supported datasets:")
    for k in sorted(supported):
        if k in installed:
            click.echo(f'- {k}: "{installed[k]}"')
        else:
            click.echo(f"* {k}: datadir.{k} (not set)")


@dataset.command(
    epilog="""Examples:

\b
  1. Check if all files of the Montgomery dataset can be loaded:

     .. code:: sh

        deepdraw dataset check -vv montgomery

\b
  2. Check if all files of multiple installed datasets can be loaded:

     .. code:: sh

        deepdraw dataset check -vv montgomery shenzhen

\b
  3. Check if all files of all installed datasets can be loaded:

     .. code:: sh

        deepdraw dataset check

""",
)
@click.argument(
    "dataset",
    nargs=-1,
)
@click.option(
    "--limit",
    "-l",
    help="Limit check to the first N samples in each dataset, making the "
    "check sensibly faster.  Set it to zero to check everything.",
    required=True,
    type=click.IntRange(0),
    default=0,
)
@verbosity_option(logger=logger, expose_value=False)
def check(dataset, limit):
    """Checks file access on one or more datasets."""
    import importlib

    to_check = _get_installed_datasets()
    supported = _get_supported_datasets()
    dataset = set(dataset)

    if dataset:
        assert supported.issuperset(
            dataset
        ), f"Unsupported datasets: {dataset-supported}"
    else:
        dataset = supported

    if dataset:
        delete = [k for k in to_check.keys() if k not in dataset]
        for k in delete:
            del to_check[k]

    if not to_check:
        click.secho(
            "WARNING: No configured datasets matching specifications",
            fg="yellow",
            bold=True,
        )
        click.echo(
            "Try deepdraw dataset list --help to get help in "
            "configuring a dataset"
        )
    else:
        errors = 0
        for k in to_check.keys():
            click.echo(f'Checking "{k}" dataset...')
            module = importlib.import_module(f"...data.{k}", __name__)
            errors += module.dataset.check(limit)
        if not errors:
            click.echo("No errors reported")
