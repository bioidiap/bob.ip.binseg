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
def train_analysis(output_folder, batch_size, **kwargs):
    """
    Analyzes the training logs for loss evolution and resource utilisation
    """
    av_loss = "average_loss"
    val_av_loss = "validation_average_loss"
    path = output_folder + "/trainlog.csv"
    pdf_path = output_folder + "/trainlog.pdf"
    trainlog_csv = pd.read_csv(path)
    plot_(trainlog_csv, "epoch", av_loss, label=av_loss)
    plot_(trainlog_csv, "epoch", val_av_loss, label=val_av_loss)
    plt.title("Trainlog analyser", y=-0.01)
    columns = list(trainlog_csv.columns)
    suptitle = "batch size = " + str(batch_size)

    if "gpu_percent" in columns:
        mean_gpu_percent = np.mean(trainlog_csv["gpu_percent"])
        suptitle += "\n Gpu : " + str(float("{:.2f}".format(mean_gpu_percent)))

    if "gpu_memory_percent" in columns:
        mean_gpu_memory_percent = np.mean(trainlog_csv["gpu_memory_percent"])
        suptitle += "%, Gpu ram : " + str(
            float("{:.2f}".format(mean_gpu_memory_percent))
        )
        +"%"

    epoch_with_best_validation = trainlog_csv["epoch"][
        np.argmin(trainlog_csv["validation_average_loss"])
    ]
    suptitle += "\n Epoch with best validation = " + str(
        epoch_with_best_validation
    )

    plt.axvline(x=epoch_with_best_validation, color="red", label="best_model")
    plt.legend()

    plt.suptitle(suptitle)

    plt.savefig(pdf_path)
    plt.clf()