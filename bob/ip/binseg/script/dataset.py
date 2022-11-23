import importlib
import logging
import os

import click
import pkg_resources

from bob.extension import rc
from bob.extension.scripts.click_helper import AliasedGroup, verbosity_option

logger = logging.getLogger(__name__)


def _get_supported_datasets():
    """Returns a list of supported dataset names"""

    basedir = pkg_resources.resource_filename(__name__, "")
    basedir = os.path.join(os.path.dirname(basedir), "data")

    retval = []
    for k in os.listdir(basedir):
        candidate = os.path.join(basedir, k)
        if os.path.isdir(candidate) and "__init__.py" in os.listdir(candidate):
            retval.append(k)
    return retval


def _get_installed_datasets():
    """Returns a list of installed datasets as regular expressions

    Returns:

        dict: A dictionary mapping dataset slugs (python-encoded names) to the
            full path of the variables containing their base directory.

    """
    base_config = rc.get("bob.ip.binseg", {})
    return {
        k: f"bob.ip.binseg.{k}.datadir"
        for k in base_config.keys()
        if "datadir" in base_config[k]
    }


@click.group(cls=AliasedGroup)
def dataset():
    """Commands for listing and verifying datasets"""
    pass


@dataset.command(
    epilog="""Examples:

\b
    1. To install a dataset, set up its data directory ("datadir").  For
       example, to setup access to DRIVE files you downloaded locally at
       the directory "/path/to/drive/files", do the following:
\b
       $ bob config set "bob.ip.binseg.drive.datadir" "/path/to/drive/files"

       Notice this setting **is** case-sensitive.

    2. List all raw datasets supported (and configured):

       $ bob binseg dataset list

""",
)
@verbosity_option()
def list(**kwargs):
    """Lists all supported and configured datasets"""

    supported = _get_supported_datasets()
    installed = _get_installed_datasets()

    click.echo("Supported datasets:")
    for k in supported:
        if k in installed:
            click.echo(f'- {k}: {installed[k]} = "{rc.get(installed[k])}"')
        else:
            click.echo(f"* {k}: bob.ip.binseg.{k}.datadir (not set)")


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
        to_check = {k: v for k, v in to_check.items() if k in dataset}

    if not to_check:
        click.echo("No configured datasets matching specifications")
        click.echo(
            "Try bob binseg dataset list --help to get help in "
            "configuring a dataset"
        )
    else:
        errors = 0
        for k in to_check:
            click.echo(f'Checking "{k}" dataset...')
            module = importlib.import_module(f"...data.{k}", __name__)
            errors += module.dataset.check(limit)
        if not errors:
            click.echo("No errors reported")
