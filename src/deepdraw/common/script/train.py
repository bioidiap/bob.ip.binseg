# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import multiprocessing
import sys

import torch

from torch.utils.data import DataLoader

from ..utils.checkpointer import Checkpointer
from .common import set_seeds, setup_pytorch_device
from ...binseg.engine.trainer import run

logger = logging.getLogger(__name__)


# def _collate_fn(batch):
#     return tuple(zip(*batch))


def base_train(
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
    detection,
    verbose,
    **kwargs,
):
    """Create base function for training segmentation / detection task."""

    device = setup_pytorch_device(device)

    set_seeds(seed, all_gpus=False)

    use_dataset = dataset
    validation_dataset = None
    extra_validation_datasets = []
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

        if "__extra_valid__" in dataset:
            if not isinstance(dataset["__extra_valid__"], list):
                raise RuntimeError(
                    f"If present, dataset['__extra_valid__'] must be a list, "
                    f"but you passed a {type(dataset['__extra_valid__'])}, "
                    f"which is invalid."
                )
            logger.info(
                f"Found {len(dataset['__extra_valid__'])} extra validation "
                f"set(s) to be tracked during training"
            )
            logger.info(
                "Extra validation sets are NOT used for model checkpointing!"
            )
            extra_validation_datasets = dataset["__extra_valid__"]

    # PyTorch dataloader
    multiproc_kwargs = dict()
    if parallel < 0:
        multiproc_kwargs["num_workers"] = 0
    else:
        multiproc_kwargs["num_workers"] = (
            parallel or multiprocessing.cpu_count()
        )

    if multiproc_kwargs["num_workers"] > 0 and sys.platform.startswith(
        "darwin"
    ):
        multiproc_kwargs[
            "multiprocessing_context"
        ] = multiprocessing.get_context("spawn")

    batch_chunk_size = batch_size
    if batch_size % batch_chunk_count != 0:
        # batch_size must be divisible by batch_chunk_count.
        raise RuntimeError(
            f"--batch-size ({batch_size}) must be divisible by "
            f"--batch-chunk-size ({batch_chunk_count})."
        )
    else:
        batch_chunk_size = batch_size // batch_chunk_count

    data_loader = DataLoader(
        dataset=use_dataset,
        batch_size=batch_chunk_size,
        shuffle=True,
        drop_last=drop_incomplete_batch,
        pin_memory=torch.cuda.is_available(),
        **multiproc_kwargs,
    )

    valid_loader = None
    if validation_dataset is not None:
        valid_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=batch_chunk_size,
            shuffle=False,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            **multiproc_kwargs,
        )

    extra_valid_loaders = [
        DataLoader(
            dataset=k,
            batch_size=batch_chunk_size,
            shuffle=False,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            **multiproc_kwargs,
        )
        for k in extra_validation_datasets
    ]

    checkpointer = Checkpointer(model, optimizer, scheduler, path=output_folder)

    arguments = {}
    arguments["epoch"] = 0
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)
    arguments["max_epoch"] = epochs

    logger.info("Training for {} epochs".format(arguments["max_epoch"]))
    logger.info("Continuing from epoch {}".format(arguments["epoch"]))

    run(
        model,
        data_loader,
        valid_loader,
        extra_valid_loaders,
        optimizer,
        criterion,
        scheduler,
        checkpointer,
        checkpoint_period,
        device,
        arguments,
        output_folder,
        monitoring_interval,
        batch_chunk_count,
    )
