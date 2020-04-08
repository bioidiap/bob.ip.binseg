#!/usr/bin/env python
# coding=utf-8

import click
from click_plugins import with_plugins

from torch.utils.data import DataLoader

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
    AliasedGroup,
)

from ..engine.evaluator import run

import logging
logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Runs evaluation on an existing dataset configuration:
\b
       $ bob binseg evaluate -vv m2unet drive-test --predictions-folder=path/to/predictions --output-folder=path/to/results
\b
    2. To run evaluation on a folder with your own images and annotations, you
       must first specify resizing, cropping, etc, so that the image can be
       correctly input to the model.  Failing to do so will likely result in
       poor performance.  To figure out such specifications, you must consult
       the dataset configuration used for **training** the provided model.
       Once you figured this out, do the following:
\b
       $ bob binseg config copy csv-dataset-example mydataset.py
       # modify "mydataset.py" to your liking
       $ bob binseg evaluate -vv m2unet mydataset.py --predictions-folder=path/to/predictions --output-folder=path/to/results
""",
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store the analysis result (created if does not exist)",
    required=True,
    default="results",
    cls=ResourceOption,
)
@click.option(
    "--predictions-folder",
    "-p",
    help="Path where predictions are currently stored",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="A torch.utils.data.dataset.Dataset instance implementing a dataset to be used for evaluating predictions, possibly including all pre-processing pipelines required",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--overlayed",
    "-A",
    help="Creates overlayed representations of the output probability maps, "
    "similar to --overlayed in prediction-mode, except it includes "
    "distinctive colours for true and false positives and false negatives.  "
    "If not set, or empty then do **NOT** output overlayed images.  "
    "Otherwise, the parameter represents the name of a folder where to "
    "store those",
    show_default=True,
    default=None,
    required=False,
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def evaluate(output_folder, predictions_folder, dataset, overlayed, **kwargs):
    """Evaluates an FCN on a binary segmentation task.
    """
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            pin_memory=False)
    run(dataset, predictions_folder, output_folder)
