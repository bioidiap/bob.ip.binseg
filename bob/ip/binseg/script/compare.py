#!/usr/bin/env python
# coding=utf-8

import click

from bob.extension.scripts.click_helper import (
    verbosity_option,
    AliasedGroup,
)

from ..utils.plot import combined_precision_recall_f1iso_confintval

import logging
logger = logging.getLogger(__name__)


@click.command(
    epilog="""Examples:

\b
    1. Compares system A and B, with their own pre-computed metric files:
\b
       $ bob binseg compare -vv A path/to/A/metrics.csv B path/to/B/metrics.csv
""",
)
@click.argument(
        'label_path',
        nargs=-1,
        )
@click.option(
    "--output",
    "-o",
    help="Path where write the output figure (PDF format)",
    show_default=True,
    required=True,
    default="comparison.pdf",
    type=click.Path(),
)
@verbosity_option()
def compare(label_path, output, **kwargs):
    """Compares multiple systems together"""

    # hack to get a dictionary from arguments passed to input
    if len(label_path) % 2 != 0:
        raise click.ClickException("Input label-paths should be doubles"
                " composed of name-path entries")
    data = dict(zip(label_path[::2], label_path[1::2]))

    fig = combined_precision_recall_f1iso_confintval(data)
    logger.info(f"Saving plot at {output}")
    fig.savefig(output)
