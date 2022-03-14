#!/usr/bin/env python
# coding=utf-8

import logging
import os

import click
import matplotlib.pyplot as plt
import numpy
import pandas

from bob.extension.scripts.click_helper import (
    ConfigCommand,
    ResourceOption,
    verbosity_option,
)

logger = logging.getLogger(__name__)


def plot_(df, x, y, label):
    plt.plot(df[x].values, df[y].values, label=label)


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Analyzes a training log and produces various plots:

       $ bob binseg train-analysis -vv --batch-size=16 log.csv

""",
)
@click.argument(
    "log",
    type=click.Path(dir_okay=False, exists=True),
)
@click.option(
    "--batch-size",
    "-b",
    help="Number of samples in every batch.",
    required=True,
    type=click.IntRange(min=1),
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
def train_analysis(log, batch_size, output_pdf, verbose, **kwargs):
    """
    Analyzes the training logs for loss evolution and resource utilisation
    """

    av_loss = "average_loss"
    val_av_loss = "validation_average_loss"

    trainlog_csv = pandas.read_csv(log)

    plot_(trainlog_csv, "epoch", av_loss, label=av_loss)
    plot_(trainlog_csv, "epoch", val_av_loss, label=val_av_loss)

    columns = list(trainlog_csv.columns)

    title = ""
    if batch_size is not None:
        title += f"batch:{batch_size}"

    if "gpu_percent" in columns:
        mean_gpu_percent = numpy.mean(trainlog_csv["gpu_percent"])
        title += f" | GPU: {mean_gpu_percent:.0f}%"

    if "gpu_memory_percent" in columns:
        mean_gpu_memory_percent = numpy.mean(trainlog_csv["gpu_memory_percent"])
        title += f" | GPU-mem: {mean_gpu_memory_percent:.0f}%"

    epoch_with_best_validation = trainlog_csv["epoch"][
        numpy.argmin(trainlog_csv["validation_average_loss"])
    ]

    plt.axvline(
        x=epoch_with_best_validation, color="red", label="lowest validation"
    )

    plt.suptitle("Trainlog analysis")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(alpha=0.6)
    plt.tight_layout()

    # makes sure the directory to save the output PDF is there
    dirname = os.path.dirname(os.path.realpath(output_pdf))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    plt.savefig(output_pdf)
