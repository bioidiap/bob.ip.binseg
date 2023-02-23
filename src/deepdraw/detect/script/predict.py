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
    entry_point_group="detect.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Runs prediction on an existing dataset configuration:
\b
       $ detect predict -vv faster_rcnn jsrt --weight=path/to/model_final_epoch.pth --output-folder=path/to/predictions
\b
    2. To run prediction on a folder with your own images, you must first
       specify resizing, cropping, etc, so that the image can be correctly
       input to the model.  Failing to do so will likely result in poor
       performance.  To figure out such specifications, you must consult the
       dataset configuration used for **training** the provided model.  Once
       you figured this out, do the following:
\b
       $ detect config copy csv-dataset-example mydataset.py
       # modify "mydataset.py" to include the base path and required transforms
       $ detect predict -vv m2unet mydataset.py --weight=path/to/model_final_epoch.pth --output-folder=path/to/predictions
""",
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store the predictions (created if does not exist)",
    required=True,
    default="results",
    cls=ResourceOption,
    type=click.Path(),
)
@click.option(
    "--model",
    "-m",
    help="A torch.nn.Module instance implementing the network to be evaluated",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="A torch.utils.data.dataset.Dataset instance implementing a dataset "
    "to be used for running prediction, possibly including all pre-processing "
    "pipelines required or, optionally, a dictionary mapping string keys to "
    "torch.utils.data.dataset.Dataset instances.  All keys that do not start "
    "with an underscore (_) will be processed.",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--batch-size",
    "-b",
    help="Number of samples in every batch (this parameter affects memory requirements for the network)",
    required=True,
    show_default=True,
    default=1,
    type=click.IntRange(min=1),
    cls=ResourceOption,
)
@click.option(
    "--device",
    "-d",
    help='A string indicating the device to use (e.g. "cpu" or "cuda:0")',
    show_default=True,
    required=True,
    default="cpu",
    cls=ResourceOption,
)
@click.option(
    "--weight",
    "-w",
    help="Path or URL to pretrained model file (.pth extension)",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--overlayed",
    "-O",
    help="Creates overlayed representations of the output bounding boxes on "
    "top of input images (store results as PNG files).   If not set, or empty "
    "then do **NOT** output overlayed images.  Otherwise, the parameter "
    "represents the name of a folder where to store those",
    show_default=True,
    default=None,
    required=False,
    cls=ResourceOption,
)
@click.option(
    "--parallel",
    "-P",
    help="""Use multiprocessing for data loading: if set to -1 (default),
    disables multiprocessing data loading.  Set to 0 to enable as many data
    loading instances as processing cores as available in the system.  Set to
    >= 1 to enable that many multiprocessing instances for data loading.""",
    type=click.IntRange(min=-1),
    show_default=True,
    required=True,
    default=-1,
    cls=ResourceOption,
)
@verbosity_option(logger=logger, cls=ResourceOption)
@click.pass_context
def predict(
    ctx,
    output_folder,
    model,
    dataset,
    batch_size,
    device,
    weight,
    overlayed,
    parallel,
    verbose,
    **kwargs,
):
    """Predicts bounding boxes for specified objects on input images."""
    from ...common.script.predict import base_predict

    ctx.invoke(
        base_predict,
        output_folder=output_folder,
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        weight=weight,
        overlayed=overlayed,
        parallel=parallel,
        detection=True,
        verbose=verbose,
    )