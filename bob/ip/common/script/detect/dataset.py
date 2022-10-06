#!/usr/bin/env python
# coding=utf-8

import importlib
import logging
import os

from pathlib import Path

import click
import pkg_resources

from bob.extension import rc
from bob.extension.scripts.click_helper import AliasedGroup, verbosity_option

logger = logging.getLogger(__name__)


def _get_supported_datasets():
    """Returns a list of supported dataset names"""

    basedir = pkg_resources.resource_filename(__name__, "")
    basedir = Path(basedir).resolve().parents[2]
    basedir = os.path.join(basedir, "detect", "data")

    retval = []
    for k in os.listdir(basedir):
        candidate = os.path.join(basedir, k)
        if os.path.isdir(candidate) and "__init__.py" in os.listdir(candidate):
            retval.append(k)
    return retval


def _get_installed_datasets():
    """Returns a list of installed datasets as regular expressions

    * group(0): the name of the key for the dataset directory
    * group("name"): the short name for the dataset

    """

    import re

    dataset_re = re.compile(r"^bob\.ip\.detect\.(?P<name>[^\.]+)\.datadir$")
    return [dataset_re.match(k) for k in rc.keys() if dataset_re.match(k)]


@click.group(cls=AliasedGroup)
def dataset():
    """Commands for listing and verifying datasets"""
    pass


@dataset.command(
    epilog="""Examples:

\b
    1. To install a dataset, set up its data directory ("datadir").  For
       example, to setup access to JSRT files you downloaded locally at
       the directory "/path/to/jsrt/files", do the following:
\b
       $ bob config set "bob.ip.detect.jsrt.datadir" "/path/to/jsrt/files"

       Notice this setting **is** case-sensitive.

    2. List all raw datasets supported (and configured):

       $ bob detect dataset list

""",
)
@verbosity_option()
def list(**kwargs):
    """Lists all supported and configured datasets"""

    supported = _get_supported_datasets()
    installed = _get_installed_datasets()
    installed = dict((k.group("name"), k.group(0)) for k in installed)

    click.echo("Supported datasets:")
    for k in supported:
        if k in installed:
            click.echo(f'- {k}: {installed[k]} = "{rc.get(installed[k])}"')
        else:
            click.echo(f"* {k}: bob.ip.detect.{k}.datadir (not set)")


@dataset.command(
    epilog="""Examples:

    1. Check if all files of the JSRT dataset can be loaded:

       $ bob detect dataset check -vv jsrt

    2. Check if all files of multiple installed datasets can be loaded:

       $ bob detect dataset check -vv jsrt shenzhen

    3. Check if all files of all installed datasets can be loaded:

       $ bob detect dataset check
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
@verbosity_option()
def check(dataset, limit, **kwargs):
    """Checks file access on one or more datasets"""

    to_check = _get_installed_datasets()

    if dataset:  # check only some
        to_check = [k for k in to_check if k.group("name") in dataset]

    if not to_check:
        click.echo("No configured datasets matching specifications")
        click.echo(
            "Try bob detect dataset list --help to get help in "
            "configuring a dataset"
        )
    else:
        errors = 0
        for k in to_check:
            click.echo(f"Checking \"{k.group('name')}\" dataset...")
            module = importlib.import_module(
                f".....detect.data.{k.group('name')}", __name__
            )
            errors += module.dataset.check(limit)
        if not errors:
            click.echo("No errors reported")
