#!/usr/bin/env python
# coding=utf-8

import os

import click
import torch
from torch.utils.data import DataLoader

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
)

from ..utils.checkpointer import Checkpointer
from .binseg import setup_pytorch_device, set_seeds

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
    "--output-folder",
    "-o",
    help="Path where to store the generated model (created if does not exist)",
    required=True,
    type=click.Path(),
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
    help="A torch.utils.data.dataset.Dataset instance implementing a dataset "
    "to be used for training the model, possibly including all pre-processing "
    "pipelines required or, optionally, a dictionary mapping string keys to "
    "torch.utils.data.dataset.Dataset instances.  At least one key "
    "named ``train`` must be available.  This dataset will be used for "
    "training the network model.  The dataset description must include all "
    "required pre-processing, including eventual data augmentation.  If a "
    "dataset named ``__train__`` is available, it is used prioritarily for "
    "training instead of ``train``.  If a dataset named ``__valid__`` is "
    "available, it is used for model validation (and automatic check-pointing) "
    "at each epoch.",
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
    help="A loss function to compute the FCN error for every sample "
    "respecting the PyTorch API for loss functions (see torch.nn.modules.loss)",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--scheduler",
    help="A learning rate scheduler that drives changes in the learning "
    "rate depending on the FCN state (see torch.optim.lr_scheduler)",
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
@click.option(
    "--drop-incomplete-batch/--no-drop-incomplete-batch",
    "-D",
    help="If set, then may drop the last batch in an epoch, in case it is "
    "incomplete.  If you set this option, you should also consider "
    "increasing the total number of epochs of training, as the total number "
    "of training steps may be reduced",
    required=True,
    show_default=True,
    default=False,
    cls=ResourceOption,
)
@click.option(
    "--epochs",
    "-e",
    help="Number of epochs (complete training set passes) to train for",
    show_default=True,
    required=True,
    default=1000,
    type=click.IntRange(min=1),
    cls=ResourceOption,
)
@click.option(
    "--checkpoint-period",
    "-p",
    help="Number of epochs after which a checkpoint is saved. "
    "A value of zero will disable check-pointing. If checkpointing is "
    "enabled and training stops, it is automatically resumed from the "
    "last saved checkpoint if training is restarted with the same "
    "configuration.",
    show_default=True,
    required=True,
    default=0,
    type=click.IntRange(min=0),
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
    type=click.IntRange(min=0),
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
    type=click.IntRange(min=0),
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def train(
    model,
    optimizer,
    scheduler,
    output_folder,
    epochs,
    batch_size,
    drop_incomplete_batch,
    criterion,
    dataset,
    checkpoint_period,
    device,
    seed,
    ssl,
    rampup,
    verbose,
    **kwargs,
):
    """Trains an FCN to perform binary segmentation

    Training is performed for a configurable number of epochs, and generates at
    least a final_model.pth.  It may also generate a number of intermediate
    checkpoints.  Checkpoints are model files (.pth files) that are stored
    during the training and useful to resume the procedure in case it stops
    abruptly.
    """

    device = setup_pytorch_device(device)

    set_seeds(seed, all_gpus=False)

    use_dataset = dataset
    validation_dataset = None
    if isinstance(dataset, dict):
        if "__train__" in dataset:
            logger.info("Found (dedicated) '__train__' set for training")
            use_dataset = dataset["__train__"]
        else:
            use_dataset = dataset["train"]

        if "__valid__" in dataset:
            logger.info("Found (dedicated) '__valid__' set for validation")
            logger.info("Will checkpoint lowest loss model on validation set")
            validation_dataset = dataset["__valid__"]

    # PyTorch dataloader
    data_loader = DataLoader(
        dataset=use_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_incomplete_batch,
        pin_memory=torch.cuda.is_available(),
    )

    valid_loader = None
    if validation_dataset is not None:
        valid_loader = DataLoader(
                dataset=validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=torch.cuda.is_available(),
                )

    checkpointer = Checkpointer(model, optimizer, scheduler, path=output_folder)

    arguments = {}
    arguments["epoch"] = 0
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)
    arguments["max_epoch"] = epochs

    logger.info("Training for {} epochs".format(arguments["max_epoch"]))
    logger.info("Continuing from epoch {}".format(arguments["epoch"]))

    if not ssl:
        from ..engine.trainer import run
        run(
            model,
            data_loader,
            valid_loader,
            optimizer,
            criterion,
            scheduler,
            checkpointer,
            checkpoint_period,
            device,
            arguments,
            output_folder,
        )

    else:
        from ..engine.ssltrainer import run
        run(
            model,
            data_loader,
            valid_loader,
            optimizer,
            criterion,
            scheduler,
            checkpointer,
            checkpoint_period,
            device,
            arguments,
            output_folder,
            rampup,
        )
