#!/usr/bin/env python
# coding=utf-8

import shutil
import inspect

import click
import pkg_resources

from bob.extension.scripts.click_helper import (
    verbosity_option,
    AliasedGroup,
)

import logging
logger = logging.getLogger(__name__)


@click.group(cls=AliasedGroup)
def config():
    """Commands for listing, describing and copying configuration resources"""
    pass


@config.command(
    epilog="""
\b
Examples:

\b
  1. Lists all configuration resources (type: bob.ip.binseg.config) installed:

\b
     $ bob binseg config list


\b
  2. Lists all configuration resources and their descriptions (notice this may
     be slow as it needs to load all modules once):

\b
     $ bob binseg config list -v

"""
)
@verbosity_option()
def list(verbose):
    """Lists configuration files installed"""

    entry_points = pkg_resources.iter_entry_points("bob.ip.binseg.config")
    entry_points = dict([(k.name, k) for k in entry_points])

    # all modules with configuration resources
    modules = set(
        k.module_name.rsplit(".", 1)[0] for k in entry_points.values()
    )
    keep_modules = []
    for k in sorted(modules):
        if k not in keep_modules and \
                not any(k.startswith(l) for l in keep_modules):
            keep_modules.append(k)
    modules = keep_modules

    # sort data entries by originating module
    entry_points_by_module = {}
    for k in modules:
        entry_points_by_module[k] = {}
        for name, ep in entry_points.items():
            if ep.module_name.startswith(k):
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

        print("module: %s" % (config_type,))
        for name in sorted(entry_points_by_module[config_type]):
            ep = entry_points[name]

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
    epilog="""
\b
Examples:

\b
  1. Describes the DRIVE (training) dataset configuration:

\b
     $ bob binseg config describe drive


\b
  2. Describes the DRIVE (training) dataset configuration and lists its
     contents:

\b
     $ bob binseg config describe drive -v

"""
)
@click.argument(
    "name", required=True, nargs=-1,
)
@verbosity_option()
def describe(name, verbose):
    """Describes a specific configuration file"""

    entry_points = pkg_resources.iter_entry_points("bob.ip.binseg.config")
    entry_points = dict([(k.name, k) for k in entry_points])

    for k in name:
        if k not in entry_points:
            logger.error("Cannot find configuration resource '%s'", k)
            continue
        ep = entry_points[k]
        print("Configuration: %s" % (ep.name,))
        print("Python Module: %s" % (ep.module_name,))
        print("")
        mod = ep.load()

        if verbose >= 1:
            fname = inspect.getfile(mod)
            print("Contents:")
            with open(fname, "r") as f:
                print(f.read())
        else:  #only output documentation
            print("Documentation:")
            print(inspect.getdoc(mod))


@config.command(
    epilog="""
\b
Examples:

\b
  1. Makes a copy of one of the stock configuration files locally, so it can be
     adapted:

\b
     $ bob binseg config copy drive -vvv newdataset.py


"""
)
@click.argument(
    "source", required=True, nargs=1,
)
@click.argument(
    "destination", required=True, nargs=1,
)
@verbosity_option()
def copy(source, destination, verbose):
    """Copies a specific configuration resource so it can be modified locally"""

    entry_points = pkg_resources.iter_entry_points("bob.ip.binseg.config")
    entry_points = dict([(k.name, k) for k in entry_points])

    if source not in entry_points:
        logger.error("Cannot find configuration resource '%s'", source)
        return 1
    ep = entry_points[source]
    mod = ep.load()
    src_name = inspect.getfile(mod)
    logger.info('cp %s -> %s' % (src_name, destination))
    shutil.copyfile(src_name, destination)
