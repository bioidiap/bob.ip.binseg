# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os

import click
import matplotlib.pyplot as plt
import numpy
import pandas

from clapper.click import ConfigCommand, ResourceOption, verbosity_option
from clapper.logging import setup
from matplotlib.backends.backend_pdf import PdfPages

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")


def _loss_evolution(df):
    """Plots the loss evolution over time (epochs)

    Parameters
    ----------

    df : pandas.DataFrame
        dataframe containing the training logs


    Returns
    -------

    figure : matplotlib.figure.Figure
        figure to be displayed or saved to file
    """

    figure = plt.figure()
    axes = figure.gca()

    axes.plot(df.epoch.values, df.loss.values, label="Training")
    if "validation_loss" in df.columns:
        axes.plot(
            df.epoch.values, df.validation_loss.values, label="Validation"
        )
        # shows a red dot on the location with the minima on the validation set
        lowest_index = numpy.argmin(df["validation_loss"])

        axes.plot(
            df.epoch.values[lowest_index],
            df.validation_loss[lowest_index],
            "mo",
            label=f"Lowest validation ({df.validation_loss[lowest_index]:.3f}@{df.epoch[lowest_index]})",
        )

    if "extra_validation_losses" in df.columns:
        # These losses are in array format. So, we read all rows, then create a
        # 2d array.  We transpose the array to iterate over each column and
        # plot the losses individually.  They are numbered from 1.
        df["extra_validation_losses"] = df["extra_validation_losses"].apply(
            lambda x: numpy.fromstring(x.strip("[]"), sep=" ")
        )
        losses = numpy.vstack(df.extra_validation_losses.values).T
        for n, k in enumerate(losses):
            axes.plot(df.epoch.values, k, label=f"Extra validation {n+1}")

    axes.set_title("Loss over time")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")

    axes.legend(loc="best")
    axes.grid(alpha=0.3)
    figure.set_layout_engine("tight")

    return figure


def _hardware_utilisation(df, const):
    """Plot the CPU utilisation over time (epochs).

    Parameters
    ----------

    df : pandas.DataFrame
        dataframe containing the training logs

    const : dict
        training and hardware constants


    Returns
    -------

    figure : matplotlib.figure.Figure
        figure to be displayed or saved to file
    """
    figure = plt.figure()
    axes = figure.gca()

    cpu_percent = df.cpu_percent.values / const["cpu_count"]
    cpu_memory = 100 * df.cpu_rss / const["cpu_memory_total"]

    axes.plot(
        df.epoch.values,
        cpu_percent,
        label=f"CPU usage (cores: {const['cpu_count']})",
    )
    axes.plot(
        df.epoch.values,
        cpu_memory,
        label=f"CPU memory (total: {const['cpu_memory_total']:.1f} Gb)",
    )
    if "gpu_percent" in df:
        axes.plot(
            df.epoch.values,
            df.gpu_percent.values,
            label=f"GPU usage (type: {const['gpu_name']})",
        )
    if "gpu_memory_percent" in df:
        axes.plot(
            df.epoch.values,
            df.gpu_memory_percent.values,
            label=f"GPU memory (total: {const['gpu_memory_total']:.1f} Gb)",
        )
    axes.set_title("Hardware utilisation over time")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Relative utilisation (%)")
    axes.set_ylim([0, 100])

    axes.legend(loc="best")
    axes.grid(alpha=0.3)
    figure.set_layout_engine("tight")

    return figure


@click.command(
    entry_point_group="deepdraw.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
  1. Analyzes a training log and produces various plots:

     .. code:: sh

        $ deepdraw train-analysis -vv log.csv constants.csv
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

    constants = pandas.read_csv(constants)
    constants = dict(zip(constants.keys(), constants.values[0]))
    data = pandas.read_csv(log)

    # makes sure the directory to save the output PDF is there
    dirname = os.path.dirname(os.path.realpath(output_pdf))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # now, do the analysis
    with PdfPages(output_pdf) as pdf:
        figure = _loss_evolution(data)
        pdf.savefig(figure)
        plt.close(figure)

        figure = _hardware_utilisation(data, constants)
        pdf.savefig(figure)
        plt.close(figure)
