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
import tabulate

from clapp.click import verbosity_option
from clapp.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")


@click.command(
    epilog="""Examples:

\b
    1. Compares system A and B, with their own pre-computed measure files:
\b
       $ detect compare -vv A path/to/A/train.csv B path/to/B/test.csv
""",
)
@click.argument(
    "label_path",
    nargs=-1,
)
@click.option(
    "--table-format",
    "-T",
    help="The format to use for the comparison table",
    show_default=True,
    required=True,
    default="rst",
    type=click.Choice(tabulate.tabulate_formats),
)
@click.option(
    "--output-table",
    "-u",
    help="Path where write the output table. If not provided, does not write "
    "write a table to file, only to stdout.",
    required=False,
    default=None,
    type=click.Path(dir_okay=False, file_okay=True),
)
@click.option(
    "--threshold",
    "-t",
    help="This number is used to select which IoU-score to use for "
    "representing a system performance.  If not set, we report the maximum "
    "IoU-score in the set, which is equivalent to threshold selection a "
    "posteriori (biased estimator), unless the performance file being "
    "considered already was pre-tunned, and contains a 'threshold_a_priori' "
    "column which we then use to pick a threshold for the dataset. "
    "You can override this behaviour by either setting this value to a "
    "floating-point number in the range [0.0, 1.0], or to a string, naming "
    "one of the systems which will be used to calculate the threshold "
    "leading to the maximum IoU-score and then applied to all other sets.",
    default=None,
    show_default=False,
    required=False,
)
@verbosity_option(
    logger=logger,
)
@click.pass_context
def compare(
    ctx,
    label_path,
    output_figure,
    table_format,
    output_table,
    threshold,
    plot_limits,
    verbose,
    **kwargs,
):
    """Compare multiple systems together."""
    from ...common.script.compare import base_compare

    ctx.invoke(
        base_compare,
        label_path=label_path,
        output_figure=output_figure,
        output_table=output_table,
        threshold=threshold,
        plot_limits=plot_limits,
        detection=True,
        verbose=verbose,
    )
