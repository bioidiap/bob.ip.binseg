#!/usr/bin/env python
# coding=utf-8

import logging
import multiprocessing
import os
import sys

import click
import torch

from torch.utils.data import DataLoader

from bob.extension.scripts.click_helper import (
    ConfigCommand,
    ResourceOption,
    verbosity_option,
)

from ...common.utils.checkpointer import Checkpointer
from .detect import download_to_tempfile, setup_pytorch_device

logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.ip.detect.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Runs prediction on an existing dataset configuration:
\b
       $ bob detect predict -vv faster_rcnn jsrt --weight=path/to/model_final_epoch.pth --output-folder=path/to/predictions
\b
    2. To run prediction on a folder with your own images, you must first
       specify resizing, cropping, etc, so that the image can be correctly
       input to the model.  Failing to do so will likely result in poor
       performance.  To figure out such specifications, you must consult the
       dataset configuration used for **training** the provided model.  Once
       you figured this out, do the following:
\b
       $ bob detect config copy csv-dataset-example mydataset.py
       # modify "mydataset.py" to include the base path and required transforms
       $ bob detect predict -vv m2unet mydataset.py --weight=path/to/model_final_epoch.pth --output-folder=path/to/predictions
""",
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store the predictions (created if does not exist)",
    required=True,
    default="results",
    cls=ResourceOption,
    type=click.Path(),
)
@click.option(
    "--model",
    "-m",
    help="A torch.nn.Module instance implementing the network to be evaluated",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="A torch.utils.data.dataset.Dataset instance implementing a dataset "
    "to be used for running prediction, possibly including all pre-processing "
    "pipelines required or, optionally, a dictionary mapping string keys to "
    "torch.utils.data.dataset.Dataset instances.  All keys that do not start "
    "with an underscore (_) will be processed.",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--batch-size",
    "-b",
    help="Number of samples in every batch (this parameter affects memory requirements for the network)",
    required=True,
    show_default=True,
    default=1,
    type=click.IntRange(min=1),
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
    "--weight",
    "-w",
    help="Path or URL to pretrained model file (.pth extension)",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--overlayed",
    "-O",
    help="Creates overlayed representations of the output bounding boxes on "
    "top of input images (store results as PNG files).   If not set, or empty "
    "then do **NOT** output overlayed images.  Otherwise, the parameter "
    "represents the name of a folder where to store those",
    show_default=True,
    default=None,
    required=False,
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
@verbosity_option(cls=ResourceOption)
def predict(
    output_folder,
    model,
    dataset,
    batch_size,
    device,
    weight,
    overlayed,
    parallel,
    **kwargs,
):
    """Predicts bounding boxes for specified objects on input images"""

    device = setup_pytorch_device(device)

    dataset = dataset if isinstance(dataset, dict) else dict(test=dataset)

    if weight.startswith("http"):
        logger.info(f"Temporarily downloading '{weight}'...")
        f = download_to_tempfile(weight, progress=True)
        weight_fullpath = os.path.abspath(f.name)
    else:
        weight_fullpath = os.path.abspath(weight)

    checkpointer = Checkpointer(model)
    checkpointer.load(weight_fullpath)

    # clean-up the overlayed path
    if overlayed is not None:
        overlayed = overlayed.strip()

    for k, v in dataset.items():

        if k.startswith("_"):
            logger.info(f"Skipping dataset '{k}' (not to be evaluated)")
            continue

        logger.info(f"Running inference on '{k}' set...")

        # PyTorch dataloader
        multiproc_kwargs = dict()
        if parallel < 0:
            multiproc_kwargs["num_workers"] = 0
        else:
            multiproc_kwargs["num_workers"] = (
                parallel or multiprocessing.cpu_count()
            )

        if multiproc_kwargs["num_workers"] > 0 and sys.platform == "darwin":
            multiproc_kwargs[
                "multiprocessing_context"
            ] = multiprocessing.get_context("spawn")

        from ..engine.predictor import run

        def _collate_fn(batch):
            return tuple(zip(*batch))

        data_loader = DataLoader(
            dataset=v,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_fn,
            **multiproc_kwargs,
        )

        run(model, data_loader, k, device, output_folder, overlayed)
