#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""The main entry for bob ip binseg (click-based) scripts."""

import pkg_resources
import click
from click_plugins import with_plugins
from bob.extension.scripts.click_helper import AliasedGroup

@with_plugins(pkg_resources.iter_entry_points("bob.ip.binseg.cli"))
@click.group(cls=AliasedGroup)
def binseg():
    """Binary 2D Image Segmentation Benchmark commands."""
