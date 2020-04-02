#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
import pkg_resources

import click
from click_plugins import with_plugins

import torch
from torch.utils.data import DataLoader

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
    AliasedGroup,
)

from ..utils.checkpointer import DetectronCheckpointer
from ..engine.trainer import do_train
from ..engine.ssltrainer import do_ssltrain

import logging
logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Trains a U-Net model (VGG-16 backbone) with DRIVE (vessel segmentation),
       on a GPU (``cuda:0``):

       $ bob binseg train -vv unet drive --batch-size=4 --device="cuda:0"

    2. Trains a HED model with HRF on a GPU (``cuda:0``):

       $ bob binseg train -vv hed hrf --batch-size=8 --device="cuda:0"

    3. Trains a M2U-Net model on the COVD-DRIVE dataset on the CPU:

       $ bob binseg train -vv m2unet covd-drive --batch-size=8

    4. Trains a DRIU model with SSL on the COVD-HRF dataset on the CPU:

       $ bob binseg train -vv --ssl driu-ssl covd-drive-ssl --batch-size=1

""",
)
@click.option(
    "--output-path",
    "-o",
    help="Path where to store the generated model (created if does not exist)",
    required=True,
    default="results",
    cls=ResourceOption,
)
@click.option(
    "--model",
    "-m",
    help="A torch.nn.Module instance implementing the network to be trained",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="A torch.utils.data.dataset.Dataset instance implementing a dataset to be used for training the model, possibly including all pre-processing pipelines required.",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--optimizer",
    help="A torch.optim.Optimizer that will be used to train the network",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--criterion",
    help="A loss function to compute the FCN error for every sample respecting the PyTorch API for loss functions (see torch.nn.modules.loss)",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--scheduler",
    help="A learning rate scheduler that drives changes in the learning rate depending on the FCN state (see torch.optim.lr_scheduler)",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--pretrained-backbone",
    "-t",
    help="URLs of a pre-trained model file that will be used to preset FCN weights (where relevant) before training starts.  (e.g. vgg-16)",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--batch-size",
    "-b",
    help="Number of samples in every batch (notice that changing this parameter affects memory requirements for the network)",
    required=True,
    show_default=True,
    default=2,
    cls=ResourceOption,
)
@click.option(
    "--epochs",
    "-e",
    help="Number of epochs used for training",
    show_default=True,
    required=True,
    default=1000,
    cls=ResourceOption,
)
@click.option(
    "--checkpoint-period",
    "-p",
    help="Number of epochs after which a checkpoint is saved",
    show_default=True,
    required=True,
    default=100,
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
    "--seed",
    "-s",
    help="Seed to use for the random number generator",
    show_default=True,
    required=False,
    default=42,
    cls=ResourceOption,
)
@click.option(
    "--ssl/--no-ssl",
    help="Switch ON/OFF semi-supervised training mode",
    show_default=True,
    required=True,
    default=False,
    cls=ResourceOption,
)
@click.option(
    "--rampup",
    "-r",
    help="Ramp-up length in epochs (for SSL training only)",
    show_default=True,
    required=True,
    default=900,
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def train(
    model,
    optimizer,
    scheduler,
    output_path,
    epochs,
    pretrained_backbone,
    batch_size,
    criterion,
    dataset,
    checkpoint_period,
    device,
    seed,
    ssl,
    rampup,
    verbose,
):
    """Trains an FCN to perform binary segmentation using a supervised approach

    Training is performed for a fixed number of steps (not configurable).
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.manual_seed(seed)

    # PyTorch dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    # Checkpointer
    checkpointer = DetectronCheckpointer(
        model, optimizer, scheduler, save_dir=output_path, save_to_disk=True
    )
    arguments = {}
    arguments["epoch"] = 0
    extra_checkpoint_data = checkpointer.load(pretrained_backbone)
    arguments.update(extra_checkpoint_data)
    arguments["max_epoch"] = epochs

    logger.info("Training for {} epochs".format(arguments["max_epoch"]))
    logger.info("Continuing from epoch {}".format(arguments["epoch"]))

    if not ssl:
        logger.info("Doing SUPERVISED training...")
        do_train(
            model,
            data_loader,
            optimizer,
            criterion,
            scheduler,
            checkpointer,
            checkpoint_period,
            device,
            arguments,
            output_path,
        )

    else:

        logger.info("Doing SEMI-SUPERVISED training...")
        do_ssltrain(
            model,
            data_loader,
            optimizer,
            criterion,
            scheduler,
            checkpointer,
            checkpoint_period,
            device,
            arguments,
            output_path,
            rampup,
        )
