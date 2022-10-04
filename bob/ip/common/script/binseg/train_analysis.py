#!/usr/bin/env python
# coding=utf-8

import logging

import click

from bob.extension.scripts.click_helper import (
    ConfigCommand,
    ResourceOption,
    verbosity_option,
)

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Analyzes a training log and produces various plots:

       $ bob binseg train-analysis -vv log.csv constants.csv

""",
)
@click.argument(
    "log",
    type=click.Path(dir_okay=False, exists=True),
)
@click.argument(
    "constants",
    type=click.Path(dir_okay=False, exists=True),
)
@click.option(
    "--output-pdf",
    "-o",
    help="Name of the output file to dump",
    required=True,
    show_default=True,
    default="trainlog.pdf",
)
@verbosity_option(cls=ResourceOption)
@click.pass_context
def train_analysis(ctx, log, constants, output_pdf, verbose, **kwargs):
    """Analyze the training logs for loss evolution and resource utilisation."""
    from ..train_analysis import base_train_analysis

    ctx.invoke(
        base_train_analysis,
        log=log,
        constants=constants,
        output_pdf=output_pdf,
        verbose=verbose,
    )
