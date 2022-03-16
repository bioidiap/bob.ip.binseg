#!/usr/bin/env python
# coding=utf-8

import logging
import os

import click
import matplotlib.pyplot as plt
import numpy
import pandas

from matplotlib.backends.backend_pdf import PdfPages

from bob.extension.scripts.click_helper import (
    ConfigCommand,
    ResourceOption,
    verbosity_option,
)

logger = logging.getLogger(__name__)


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

    axes.plot(df.epoch.values, df.average_loss.values, label="Avg. training")
    axes.plot(
        df.epoch.values,
        df.validation_average_loss.values,
        label="Avg. validation",
    )
    axes.set_title("Loss over time")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")

    # shows a red dot on the location with the minima on the validation set
    lowest_index = numpy.argmin(df["validation_average_loss"])

    axes.plot(
        df.epoch.values[lowest_index],
        df.validation_average_loss[lowest_index],
        "go",
        label=f"Lowest validation ({df.validation_average_loss[lowest_index]:g})",
    )

    axes.legend(loc="best")
    axes.grid(alpha=0.3)
    figure.set_tight_layout(None)

    return figure


def _hardware_utilisation(df, const):
    """Plots the CPU utilisation over time (epochs)

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
    figure.set_tight_layout(None)

    return figure


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
def train_analysis(log, constants, output_pdf, verbose, **kwargs):
    """
    Analyzes the training logs for loss evolution and resource utilisation
    """

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
