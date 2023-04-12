# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import math
import multiprocessing
import random
import sys

import torch

from torch.utils.data import DataLoader

from ..utils.checkpointer import Checkpointer
from .common import set_seeds, setup_pytorch_device

logger = logging.getLogger(__name__)


def _collate_fn(batch):
    return tuple(zip(*batch))


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

    def semi_supervised_use_dataset(dataset):
        logger.info("Start setting semi-supervised training dataset")
        datalist = [None] * (
            ((len(dataset["__unlabeled_train__"])) + len(dataset["train"])) * 2
        )
        dataset_dic = {"train": datalist}

        mylistun = []
        mylist = []
        for e in dataset["__unlabeled_train__"]:
            mylistun.append(e)
        for e1 in mylistun:
            e1.append("0")
        for e in dataset["train"]:
            mylist.append(e)
        for e in mylist:
            e.append("1")

        # batch size shoud be bigger than 1
        if batch_size == 1:
            raise RuntimeError(
                f"--batch-size ({batch_size}) must be lager than 1)."
            )

        elif batch_size > 2:
            k = (
                (batch_size - 1)
                * len(dataset["train"])
                // len(dataset["__unlabeled_train__"])
            )
        else:
            k = 1

            # k is how many labeled data can be allocated to one batch
            # if k is smaller than 1, the labeled data is not enough for only one in every batch. Then we need to shuffle and reuse the labeled data

            if k == 0:
                logger.info("Not enough labeled samples for all batches")
                myshuffle = random.sample(mylist, len(mylist))
                for i in range(
                    len(mylistun) // (batch_size - 1)
                ):  # i is the number of batches
                    for j in range(batch_size):
                        if j == 0:
                            if i < len(mylist):
                                dataset_dic["train"][i * batch_size] = mylist[i]
                            else:
                                dataset_dic["train"][
                                    i * batch_size
                                ] = myshuffle[i % len(mylist)]
                        else:
                            if i * (batch_size - 1) + j < len(mylistun):
                                dataset_dic["train"][
                                    i * batch_size + j
                                ] = mylistun[i * (batch_size - 1) + j]

            # if k is larger than 0, we will try to balabce the labeled data and unlabeled data in every batch. When unlabeled data is used up, the rest of the batch will be filled with labeled data.
            else:
                # j is the number of unlabeled data in one batch
                j = math.ceil(
                    len(mylistun)
                    / ((len(mylist) + len(mylistun)) // batch_size)
                )
                logger.info(
                    f"The number of unlabeled samples in one batch is {j}"
                )
                k = len(mylistun)
                for i in range(len(mylistun) + len(mylist)):
                    if k > 0:
                        if i % batch_size < j:
                            dataset_dic["train"][i] = mylistun[
                                len(mylistun) - k
                            ]
                            k = k - 1
                        else:
                            dataset_dic["train"][i] = mylist[
                                i - len(mylistun) + k
                            ]
                    else:
                        dataset_dic["train"][i] = mylist[i - len(mylistun)]

        res = [i for i in dataset_dic["train"] if i is not None]
        dataset_dic["train"] = res
        return dataset_dic

    if isinstance(dataset, dict):
        if "__unlabeled_train__" in dataset:
            logger.info(
                "Found (dedicated) 'unlabeled_train' set for semi-supervised training"
            )
            use_dataset = semi_supervised_use_dataset(dataset)["train"]
        elif "__train__" in dataset:
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

    if detection:
        from ...detect.engine.trainer import run

        data_loader = DataLoader(
            dataset=use_dataset,
            batch_size=batch_chunk_size,
            shuffle=True,
            drop_last=drop_incomplete_batch,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_fn,
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
                collate_fn=_collate_fn,
                **multiproc_kwargs,
            )

        extra_valid_loaders = [
            DataLoader(
                dataset=k,
                batch_size=batch_chunk_size,
                shuffle=False,
                drop_last=False,
                pin_memory=torch.cuda.is_available(),
                collate_fn=_collate_fn,
                **multiproc_kwargs,
            )
            for k in extra_validation_datasets
        ]
    else:
        from ...binseg.engine.trainer import run

        # In the mean teacher model, the training data is not shuffled.
        if hasattr(model, "name") and model.name == "mean_teacher":
            data_loader = DataLoader(
                dataset=use_dataset,
                batch_size=batch_chunk_size,
                shuffle=False,
                drop_last=drop_incomplete_batch,
                # set for one GPU
                pin_memory=torch.cuda.is_available(),
                # set for multiple GPUs
                # persistent_workers=True,
                # pin_memory=False,
                **multiproc_kwargs,
            )
        else:
            data_loader = DataLoader(
                dataset=use_dataset,
                batch_size=batch_chunk_size,
                shuffle=True,
                drop_last=drop_incomplete_batch,
                # set for one GPU
                pin_memory=torch.cuda.is_available(),
                # set for multiple GPUs
                # persistent_workers=True,
                # pin_memory=False,
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
