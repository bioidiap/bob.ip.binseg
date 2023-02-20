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
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Generate masks for supported dataset by binseg. Ex: refuge.
\b
       $ binseg mkmask --dataset="refuge" --globs="Training400/*Glaucoma/*.jpg" --globs="Training400/*AMD/*.jpg" --threshold=5
\b
    Or you can generate the same results with this command

\b
       $ binseg mkmask -d "refuge" -g "Training400/*Glaucoma/*.jpg" -g "Training400/*AMD/*.jpg" -t 5

\b
    2. Generate masks for non supported dataset by binseg

\b
        $ binseg mkmask -d "Path/to/dataset" -g "glob1" -g "glob2" -g glob3  -t 4


""",
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store the generated model (created if does not exist)",
    required=True,
    type=click.Path(),
    default="masks",
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="""The base path to the dataset to which we want to generate the masks. \\
    In case you have already configured the path for the datasets supported by binseg, \\
    you can just use the name of the dataset as written in the config. """,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--globs",
    "-g",
    help="""The global path to the dataset to which we want to generate the masks.\\
    We need to specify the path for the images ,\\
    Ex : --globs="images/*.jpg"\\
    It also can be used multiple time.
    """,
    required=True,
    multiple=True,
    cls=ResourceOption,
)
@click.option(
    "--threshold",
    "-t",
    help=" Generating a mask needs a threshold to be fixed in order to transform the image to binary ",
    required=True,
    cls=ResourceOption,
)
@verbosity_option(logger=logger, cls=ResourceOption)
@click.pass_context
def mkmask(ctx, dataset, globs, threshold, output_folder, verbose, **kwargs):
    """Commands for generating masks for images in a dataset."""
    from ...common.script.mkmask import base_mkmask

    ctx.invoke(
        base_mkmask,
        dataset=dataset,
        globs=globs,
        threshold=threshold,
        output_folder=output_folder,
        verbose=verbose,
    )
