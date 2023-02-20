#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import click

from clapp.click import ConfigCommand, ResourceOption, verbosity_option
from clapp.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")


@click.command(
    entry_point_group="binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Analyzes a training log and produces various plots:

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
