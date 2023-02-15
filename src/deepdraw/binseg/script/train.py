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

import logging

import click

from bob.extension.scripts.click_helper import (
    ConfigCommand,
    ResourceOption,
    verbosity_option,
)

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
    help="A dictionary mapping string keys to "
    "torch.utils.data.dataset.Dataset instances implementing datasets "
    "to be used for training and validating the model, possibly including all "
    "pre-processing pipelines required or, optionally, a dictionary mapping "
    "string keys to torch.utils.data.dataset.Dataset instances.  At least "
    "one key named ``train`` must be available.  This dataset will be used for "
    "training the network model.  The dataset description must include all "
    "required pre-processing, including eventual data augmentation.  If a "
    "dataset named ``__train__`` is available, it is used prioritarily for "
    "training instead of ``train``.  If a dataset named ``__valid__`` is "
    "available, it is used for model validation (and automatic "
    "check-pointing) at each epoch.  If a dataset list named "
    "``__extra_valid__`` is available, then it will be tracked during the "
    "validation process and its loss output at the training log as well, "
    "in the format of an array occupying a single column.  All other keys "
    "are considered test datasets and are ignored during training",
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
    "--drop-incomplete-batch is set, in which case this batch is not used.",
    required=True,
    show_default=True,
    default=2,
    type=click.IntRange(min=1),
    cls=ResourceOption,
)
@click.option(
    "--batch-chunk-count",
    "-c",
    help="Number of chunks in every batch (this parameter affects "
    "memory requirements for the network). The number of samples "
    "loaded for every iteration will be batch-size/batch-chunk-count. "
    "batch-size needs to be divisible by batch-chunk-count, otherwise an "
    "error will be raised. This parameter is used to reduce number of "
    "samples loaded in each iteration, in order to reduce the memory usage "
    "in exchange for processing time (more iterations).  This is specially "
    "interesting whe one is running with GPUs with limited RAM. The "
    "default of 1 forces the whole batch to be processed at once.  Otherwise "
    "the batch is broken into batch-chunk-count pieces, and gradients are "
    "accumulated to complete each batch.",
    required=True,
    show_default=True,
    default=1,
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
    help="Number of epochs (complete training set passes) to train for. "
    "If continuing from a saved checkpoint, ensure to provide a greater "
    "number of epochs than that saved on the checkpoint to be loaded. ",
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
@click.option(
    "--monitoring-interval",
    "-I",
    help="""Time between checks for the use of resources during each training
    epoch.  An interval of 5 seconds, for example, will lead to CPU and GPU
    resources being probed every 5 seconds during each training epoch.
    Values registered in the training logs correspond to averages (or maxima)
    observed through possibly many probes in each epoch.  Notice that setting a
    very small value may cause the probing process to become extremely busy,
    potentially biasing the overall perception of resource usage.""",
    type=click.FloatRange(min=0.1),
    show_default=True,
    required=True,
    default=5.0,
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
@click.pass_context
def train(
    ctx,
    model,
    optimizer,
    scheduler,
    output_folder,
    epochs,
    batch_size,
    batch_chunk_count,
    drop_incomplete_batch,
    criterion,
    dataset,
    checkpoint_period,
    device,
    seed,
    parallel,
    monitoring_interval,
    verbose,
    **kwargs,
):
    """Trains an FCN to perform binary segmentation.

    Training is performed for a configurable number of epochs, and generates at
    least a final_model.pth.  It may also generate a number of intermediate
    checkpoints.  Checkpoints are model files (.pth files) that are stored
    during the training and useful to resume the procedure in case it stops
    abruptly.

    Tip: In case the model has been trained over a number of epochs, it is
    possible to continue training, by simply relaunching the same command, and
    changing the number of epochs to a number greater than the number where
    the original training session stopped (or the last checkpoint was saved).
    """
    from ...common.script.train import base_train

    ctx.invoke(
        base_train,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_folder=output_folder,
        epochs=epochs,
        batch_size=batch_size,
        batch_chunk_count=batch_chunk_count,
        drop_incomplete_batch=drop_incomplete_batch,
        criterion=criterion,
        dataset=dataset,
        checkpoint_period=checkpoint_period,
        device=device,
        seed=seed,
        parallel=parallel,
        monitoring_interval=monitoring_interval,
        detection=False,
        verbose=verbose,
    )
