#!/usr/bin/env python
# coding=utf-8

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

from ..engine.predictor import run

import logging
logger = logging.getLogger(__name__)


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Runs prediction on an existing dataset configuration:

       $ bob binseg predict -vv m2unet drive-test --weight=path/to/model_final.pth --output-path=path/to/predictions
\b
    2. To run prediction on a folder with your own images, you must first
       specify resizing, cropping, etc, so that the image can be correctly
       input to the model.  Failing to do so will likely result in poor
       performance.  To figure out such specifications, you must consult the
       dataset configuration used for **training** the provided model.  Once
       you figured this out, do the following:

       $ bob binseg config copy image-folder myfolder.py
       # modify "myfolder.py" to include the base path and required transforms
       $ bob binseg predict -vv m2unet myfolder.py --weight=path/to/model_final.pth --output-path=path/to/predictions
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
def predict(output_path, model, dataset, batch_size, device, weight, **kwargs):
    """Predicts vessel map (probabilities) on input images"""

    # PyTorch dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    # checkpointer, loads pre-fit model
    checkpointer = DetectronCheckpointer(model, save_dir=output_path,
            save_to_disk=False)
    checkpointer.load(weight)

    run(model, data_loader, device, output_path)
