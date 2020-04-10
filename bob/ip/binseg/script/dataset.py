#!/usr/bin/env python
# coding=utf-8

import importlib
import click

from bob.extension.scripts.click_helper import (
    verbosity_option,
    AliasedGroup,
)


import logging
logger = logging.getLogger(__name__)


def _get_installed_datasets():
    """Returns a list of installed datasets as regular expressions

    * group(0): the name of the key for the dataset directory
    * group("name"): the short name for the dataset

    """

    import re
    from bob.extension import rc
    dataset_re = re.compile(r'^bob\.ip\.binseg\.(?P<name>[^\.]+)\.datadir$')
    return [dataset_re.match(k) for k in rc.keys() if dataset_re.match(k)]


@click.group(cls=AliasedGroup)
def dataset():
    """Commands for listing, describing and copying configuration resources"""
    pass


@dataset.command(
    epilog="""Examples:

\b
    1. To install a dataset, set up its data directory ("datadir").  For
       example, to setup access to DRIVE files you downloaded locally at
       the directory "/path/to/drive/files", do the following:
\b
       $ bob config set "bob.ip.binseg.drive.datadir" "/path/to/drive/files"

       Notice this setting is **NOT** case-insensitive.

    2. List all raw datasets available (and configured):

       $ bob binseg dataset list -vv

""",
)
@verbosity_option()
def list(**kwargs):
    """Lists all installed datasets"""

    installed = _get_installed_datasets()
    if installed:
        click.echo("Configured datasets:")
        for k in installed:
            value = bob.extension.rc.get(k.group(0))
            click.echo(f"- {k.group('name')}: {k.group(0)} = \"{value}\"")
    else:
        click.echo("No configured datasets")
        click.echo("Try --help to get help in configuring a dataset")


@dataset.command(
    epilog="""Examples:

    1. Check if all files of the DRIVE dataset can be loaded:

       $ bob binseg dataset check -vv drive

    2. Check if all files of multiple installed datasets can be loaded:

       $ bob binseg dataset check -vv drive stare

    3. Check if all files of all installed datasets can be loaded:

       $ bob binseg dataset check
""",
)
@click.argument(
        'dataset',
        nargs=-1,
        )
@verbosity_option()
def check(dataset, **kwargs):
    """Checks file access on one or more datasets"""

    to_check = _get_installed_datasets()

    if dataset:  #check only some
        to_check = [k for k in to_check if k.group("name") in dataset]

    if not dataset:
        click.echo("No configured datasets matching specifications")
        click.echo("Try bob binseg dataset list --help to get help in "
                "configuring a dataset")
    else:
        for k in to_check:
            click.echo(f"Checking \"{k.group('name')}\" dataset...")
            module = importlib.import_module(f"...data.{k.group('name')}",
                    __name__)
            module.dataset.check()
