#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""The main entry for bob ip binseg (click-based) scripts."""


import os
import pkg_resources

import click
from click_plugins import with_plugins

import logging
import torch

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
    AliasedGroup,
)

from bob.ip.binseg.utils.checkpointer import DetectronCheckpointer
from torch.utils.data import DataLoader
from bob.ip.binseg.utils.plot import plot_overview
from bob.ip.binseg.utils.click import OptionEatAll
from bob.ip.binseg.utils.rsttable import create_overview_grid
from bob.ip.binseg.utils.plot import metricsviz
from bob.ip.binseg.utils.transformfolder import transformfolder as transfld

logger = logging.getLogger(__name__)


@with_plugins(pkg_resources.iter_entry_points("bob.ip.binseg.cli"))
@click.group(cls=AliasedGroup)
def binseg():
    """Binary 2D Image Segmentation Benchmark commands."""


# Plot comparison
@binseg.command(entry_point_group="bob.ip.binseg.config", cls=ConfigCommand)
@click.option(
    "--output-path-list",
    "-l",
    required=True,
    help="Pass all output paths as arguments",
    cls=OptionEatAll,
)
@click.option(
    "--output-path", "-o", required=True,
)
@click.option(
    "--title", "-t", required=False,
)
@verbosity_option(cls=ResourceOption)
def compare(output_path_list, output_path, title, **kwargs):
    """ Compares multiple metrics files that are stored in the format mymodel/results/Metrics.csv """
    logger.debug("Output paths: {}".format(output_path_list))
    logger.info("Plotting precision vs recall curves for {}".format(output_path_list))
    fig = plot_overview(output_path_list, title)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig_filename = os.path.join(output_path, "precision_recall_comparison.pdf")
    logger.info("saving {}".format(fig_filename))
    fig.savefig(fig_filename)


# Create grid table with results
@binseg.command(entry_point_group="bob.ip.binseg.config", cls=ConfigCommand)
@click.option(
    "--output-path", "-o", required=True,
)
@verbosity_option(cls=ResourceOption)
def gridtable(output_path, **kwargs):
    """ Creates an overview table in grid rst format for all Metrics.csv in the output_path
    tree structure:
        ├── DATABASE
        ├── MODEL
            ├── images
            └── results
    """
    logger.info("Creating grid for all results in {}".format(output_path))
    create_overview_grid(output_path)


# Create metrics viz
@binseg.command(entry_point_group="bob.ip.binseg.config", cls=ConfigCommand)
@click.option("--dataset", "-d", required=True, cls=ResourceOption)
@click.option(
    "--output-path", "-o", required=True,
)
@verbosity_option(cls=ResourceOption)
def visualize(dataset, output_path, **kwargs):
    """ Creates the following visualizations of the probabilties output maps:
    overlayed: test images overlayed with prediction probabilities vessel tree
    tpfnfpviz: highlights true positives, false negatives and false positives

    Required tree structure:
    ├── DATABASE
        ├── MODEL
            ├── images
            └── results
    """
    logger.info("Creating TP, FP, FN visualizations for {}".format(output_path))
    metricsviz(dataset=dataset, output_path=output_path)

# Apply image transforms to a folder containing images
@binseg.command(entry_point_group="bob.ip.binseg.config", cls=ConfigCommand)
@click.option("--source-path", "-s", required=True, cls=ResourceOption)
@click.option("--target-path", "-t", required=True, cls=ResourceOption)
@click.option("--transforms", "-a", required=True, cls=ResourceOption)
@verbosity_option(cls=ResourceOption)
def transformfolder(source_path, target_path, transforms, **kwargs):
    logger.info(
        "Applying transforms to images in {} and saving them to {}".format(
            source_path, target_path
        )
    )
    transfld(source_path, target_path, transforms)


# Evaluate only. Runs evaluation on predicted probability maps (--prediction-folder)
@binseg.command(entry_point_group="bob.ip.binseg.config", cls=ConfigCommand)
@click.option(
    "--output-path", "-o", required=True, default="output", cls=ResourceOption
)
@click.option(
    "--prediction-folder",
    "-p",
    help="Path containing output probability maps",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--prediction-extension",
    "-x",
    help='Extension (e.g. ".png") for the prediction files',
    default=".png",
    required=False,
    cls=ResourceOption,
)
@click.option("--dataset", "-d", required=True, cls=ResourceOption)
@click.option("--title", required=False, cls=ResourceOption)
@click.option("--legend", cls=ResourceOption)
@verbosity_option(cls=ResourceOption)
def evalpred(
    output_path,
    prediction_folder,
    prediction_extension,
    dataset,
    title,
    legend,
    **kwargs
):
    """ Run inference and evalaute the model performance """

    # PyTorch dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    # Run eval
    do_eval(
        prediction_folder,
        data_loader,
        output_folder=output_path,
        title=title,
        legend=legend,
        prediction_extension=prediction_extension,
    )
