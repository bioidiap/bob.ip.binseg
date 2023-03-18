# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import click

from clapper.click import ConfigCommand, ResourceOption, verbosity_option
from clapper.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")


@click.command(
    entry_point_group="detect.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
 1. Analyzes a training log and produces various plots:

     .. code:: sh

        $ binseg train-analysis -vv log.csv constants.csv

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
@verbosity_option(logger=logger, cls=ResourceOption)
@click.pass_context
def train_analysis(ctx, log, constants, output_pdf, verbose, **kwargs):
    """Analyze the training logs for loss evolution and resource
    utilisation."""
    from ...common.script.train_analysis import base_train_analysis

    ctx.invoke(
        base_train_analysis,
        log=log,
        constants=constants,
        output_pdf=output_pdf,
        verbose=verbose,
    )
