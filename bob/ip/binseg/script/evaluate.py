#!/usr/bin/env python
# coding=utf-8

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
from ..engine.inferencer import do_inference

import logging
logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Evaluates a M2U-Net model on the DRIVE test set:

       $ bob binseg evaluate -vv m2unet drive-test --weight=results/model_final.pth

""",
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
    help="A torch.utils.data.dataset.Dataset instance implementing a dataset to be used for evaluating the model, possibly including all pre-processing pipelines required",
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
@verbosity_option(cls=ResourceOption)
def evaluate(model, output_path, device, batch_size, dataset, weight, **kwargs):
    """Evaluates an FCN on a binary segmentation task.
    """

    # PyTorch dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    # checkpointer, load last model in dir
    checkpointer = DetectronCheckpointer(
        model, save_dir=output_path, save_to_disk=False
    )
    checkpointer.load(weight)
    do_inference(model, data_loader, device, output_path)
