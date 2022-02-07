#!/usr/bin/env python
# coding=utf-8

import logging

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bob.extension.scripts.click_helper import ResourceOption, verbosity_option

logger = logging.getLogger(__name__)


def plot_(df, x, y, label):
    plt.plot(df[x].values, df[y].values, label=label)


@click.option(
    "--output-folder",
    "-o",
    help="Path where to store the generated model (created if does not exist)",
    required=True,
    type=click.Path(),
    default="results",
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="""The base path to the dataset to which we want to generate the masks. \\
    In case you have already configured the path for the datasets supported by bob, \\
    you can just use the name of the dataset as written in the config. """,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--model",
    "-m",
    help="""Model used to analyze \\ """,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--batch-size",
    "-b",
    help="Number of samples in every batch (this parameter affects "
    "memory requirements for the network).  If the number of samples in "
    "the batch is larger than the total number of samples available for "
    "training, this value is truncated.  If this number is smaller, then "
    "batches of the specified size are created and fed to the network "
    "until there are no more new samples to feed (epoch is finished).  "
    "If the total number of training samples is not a multiple of the "
    "batch-size, the last batch will be smaller than the first, unless "
    "--drop-incomplete--batch is set, in which case this batch is not used.",
    required=True,
    show_default=True,
    default=2,
    type=click.IntRange(min=1),
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def csv_analyser(dataset, model, output_folder, batch_size, **kwargs):
    """
    Run an alayze on the trainlog.csv where it creates a pdf file with a plot describing the validation and training losses.
    """
    av_loss = "average_loss"
    val_av_loss = "validation_average_loss"
    path = output_folder + "/model/trainlog.csv"
    pdf_path = output_folder + "/analyse.pdf"
    trainlog_csv = pd.read_csv(path)
    plot_(trainlog_csv, "epoch", av_loss, label=av_loss)
    plot_(trainlog_csv, "epoch", val_av_loss, label=val_av_loss)
    plt.title(model + "_" + dataset, y=-0.01)
    mean_gpu_percent = np.mean(trainlog_csv["gpu_percent"])
    mean_gpu_memory_percent = np.mean(trainlog_csv["gpu_memory_percent"])

    epoch_with_best_validation = trainlog_csv["epoch"][
        np.argmin(trainlog_csv["validation_average_loss"])
    ]

    plt.axvline(x=epoch_with_best_validation, color="red", label="best_model")
    plt.legend()
    plt.suptitle(
        "batch size = "
        + str(batch_size)
        + "\n Gpu : "
        + str(mean_gpu_percent)
        + "%,Gpu ram :"
        + str(mean_gpu_memory_percent)
        + "%"
        + "\n Epoch with best validation = "
        + str(epoch_with_best_validation)
    )
    plt.savefig(pdf_path)
    plt.clf()
