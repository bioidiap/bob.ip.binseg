# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import importlib.metadata
import inspect
import typing

import click

from clapper.click import AliasedGroup, verbosity_option
from clapper.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")


def _retrieve_entry_points(
    group: str,
) -> typing.Iterable[importlib.metadata.EntryPoint]:
    """Wraps various entry-point retrieval mechanisms.

    For Python 3.9 and 3.10,
    :py:func:`importlib.metadata.entry_points()`
    returns a dictionary keyed by entry-point group names.  From Python
    3.10
    onwards, one may pass the ``group`` keyword to that function to
    enable
    pre-filtering, or use the ``select()`` method on the returned value,
    which
    is no longer a dictionary.

    For anything before Python 3.8, you must use the backported library
    ``importlib_metadata``.
    """
    import sys

    if sys.version_info[:2] < (3, 10):
        all_entry_points = importlib.metadata.entry_points()
        return all_entry_points.get(group, [])  # Python 3.9

    # Python 3.10 and above
    return importlib.metadata.entry_points().select(group=group)


@click.group(cls=AliasedGroup)
def config():
    """Commands for listing, describing and copying configuration resources."""
    pass


@config.command(
    epilog="""Examples:

\b
  1. Lists all configuration resources (type: deepdraw.config) installed:

     .. code:: sh

        deepdraw config list


\b
  2. Lists all configuration resources and their descriptions (notice this may
     be slow as it needs to load all modules once):

     .. code:: sh

        deepdraw config list -v
"""
)
@verbosity_option(logger=logger)
def list(verbose) -> None:
    """Lists configuration files installed."""
    entry_points = _retrieve_entry_points("deepdraw.config")
    entry_point_dict = {k.name: k for k in entry_points}

    # all modules with configuration resources
    modules = {k.module.rsplit(".", 1)[0] for k in entry_point_dict.values()}
    keep_modules: set[str] = set()
    for k in sorted(modules):
        if k not in keep_modules and not any(
            k.startswith(element) for element in keep_modules
        ):
            keep_modules.add(k)
    modules = keep_modules

    # sort data entries by originating module
    entry_points_by_module: dict[str, dict[str, typing.Any]] = {}
    for k in modules:
        entry_points_by_module[k] = {}
        for name, ep in entry_point_dict.items():
            if ep.module.startswith(k):
                entry_points_by_module[k][name] = ep

    for config_type in sorted(entry_points_by_module):
        # calculates the longest config name so we offset the printing
        longest_name_length = max(
            len(k) for k in entry_points_by_module[config_type].keys()
        )

        # set-up printing options
        print_string = "  %%-%ds   %%s" % (longest_name_length,)
        # 79 - 4 spaces = 75 (see string above)
        description_leftover = 75 - longest_name_length

        print(f"module: {config_type}")
        for name in sorted(entry_points_by_module[config_type]):
            ep = entry_point_dict[name]

            if verbose >= 1:
                module = ep.load()
                doc = inspect.getdoc(module)
                if doc is not None:
                    summary = doc.split("\n\n")[0]
                else:
                    summary = "<DOCSTRING NOT AVAILABLE>"
            else:
                summary = ""

            summary = (
                (summary[: (description_leftover - 3)] + "...")
                if len(summary) > (description_leftover - 3)
                else summary
            )

            print(print_string % (name, summary))


@config.command(
    epilog="""Examples:

\b
  1. Describes the Montgomery dataset configuration:

     .. code:: sh

        deepdraw config describe montgomery


\b
  2. Describes the Montgomery dataset configuration and lists its
     contents:

     .. code:: sh

        deepdraw config describe montgomery -v

"""
)
@click.argument(
    "name",
    required=True,
    nargs=-1,
)
@verbosity_option(logger=logger)
def describe(name, verbose) -> None:
    """Describes a specific configuration file."""
    entry_points = _retrieve_entry_points("deepdraw.config")
    entry_point_dict = {k.name: k for k in entry_points}

    for k in name:
        if k not in entry_point_dict:
            logger.error("Cannot find configuration resource '%s'", k)
            continue
        ep = entry_point_dict[k]
        print(f"Configuration: {ep.name}")
        print(f"Python Module: {ep.module}")
        print("")
        mod = ep.load()

        if verbose >= 1:
            fname = inspect.getfile(mod)
            print("Contents:")
            with open(fname) as f:
                print(f.read())
        else:  # only output documentation
            print("Documentation:")
            print(inspect.getdoc(mod))


@config.command(
    epilog="""Examples:

\b
  1. Makes a copy of one of the stock configuration files locally, so it can be
     adapted:

     .. code:: sh

        $ deepdraw config copy montgomery -vvv newdataset.py

"""
)
@click.argument(
    "source",
    required=True,
    nargs=1,
)
@click.argument(
    "destination",
    required=True,
    nargs=1,
)
@verbosity_option(logger=logger, expose_value=False)
def copy(source, destination) -> None:
    """Copy a specific configuration resource so it can be modified locally."""
    import shutil

    entry_points = _retrieve_entry_points("deepdraw.config")
    entry_point_dict = {k.name: k for k in entry_points}

    if source not in entry_point_dict:
        logger.error("Cannot find configuration resource '%s'", source)
        return

    ep = entry_point_dict[source]
    mod = ep.load()
    src_name = inspect.getfile(mod)
    logger.info(f"cp {src_name} -> {destination}")
    shutil.copyfile(src_name, destination)
