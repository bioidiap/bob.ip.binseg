# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import contextlib
import csv
import datetime
import logging
import os
import shutil
import sys
import time

import numpy
import torch

from tqdm import tqdm

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer

from ..utils.accelerator import AcceleratorProcessor
from ..utils.resources import ResourceMonitor, cpu_constants, gpu_constants
from ..utils.summary import summary
from .callbacks import LoggingCallback

logger = logging.getLogger(__name__)




def save_model_summary(output_folder, model):
    """Save a little summary of the model in a txt file.

    Parameters
    ----------

    output_folder : str
        output path

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)

    Returns
    -------
    r : str
        The model summary in a text format.

    n : int
        The number of parameters of the model.
    """
    summary_path = os.path.join(output_folder, "model_summary.txt")
    logger.info(f"Saving model summary at {summary_path}...")
    with open(summary_path, "w") as f:
        r, n = summary(model)
        logger.info(f"Model has {n} parameters...")
        f.write(r)
    return r, n


def static_information_to_csv(static_logfile_name, device, n):
    """Save the static information in a csv file.

    Parameters
    ----------

    static_logfile_name : str
        The static file name which is a join between the output folder and "constant.csv"
    """
    if os.path.exists(static_logfile_name):
        backup = static_logfile_name + "~"
        if os.path.exists(backup):
            os.unlink(backup)
        shutil.move(static_logfile_name, backup)
    with open(static_logfile_name, "w", newline="") as f:
        logdata = cpu_constants()
        if device == "cuda":
            logdata += gpu_constants()
        logdata += (("model_size", n),)
        logwriter = csv.DictWriter(f, fieldnames=[k[0] for k in logdata])
        logwriter.writeheader()
        logwriter.writerow(dict(k for k in logdata))


def run(
    model,
    data_loader,
    valid_loader,
    extra_valid_loaders,
    checkpoint_period,
    accelerator,
    arguments,
    output_folder,
    monitoring_interval,
    batch_chunk_count,
    checkpoint
):
    """Fits an FCN model using supervised learning and save it to disk.

    This method supports periodic checkpointing and the output of a
    CSV-formatted log with the evolution of some figures during training.


    Parameters
    ----------

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)

    data_loader : :py:class:`torch.utils.data.DataLoader`
        To be used to train the model

    valid_loaders : :py:class:`list` of :py:class:`torch.utils.data.DataLoader`
        To be used to validate the model and enable automatic checkpointing.
        If ``None``, then do not validate it.

    extra_valid_loaders : :py:class:`list` of :py:class:`torch.utils.data.DataLoader`
        To be used to validate the model, however **does not affect** automatic
        checkpointing. If empty, then does not log anything else.  Otherwise,
        an extra column with the loss of every dataset in this list is kept on
        the final training log.

    optimizer : :py:mod:`torch.optim`

    criterion : :py:class:`torch.nn.modules.loss._Loss`
        loss function

    scheduler : :py:mod:`torch.optim`
        learning rate scheduler

    checkpointer : :py:class:`deepdraw.utils.checkpointer.Checkpointer`
        checkpointer implementation

    checkpoint_period : int
        save a checkpoint every ``n`` epochs.  If set to ``0`` (zero), then do
        not save intermediary checkpoints

    device : :py:class:`torch.device`
        device to use

    arguments : dict
        start and end epochs

    output_folder : str
        output path

    monitoring_interval : int, float
        interval, in seconds (or fractions), through which we should monitor
        resources during training.

    batch_chunk_count: int
        If this number is different than 1, then each batch will be divided in
        this number of chunks.  Gradients will be accumulated to perform each
        mini-batch.   This is particularly interesting when one has limited RAM
        on the GPU, but would like to keep training with larger batches.  One
        exchanges for longer processing times in this case.
    """

    

    max_epoch = arguments["max_epoch"]

    accelerator_processor = AcceleratorProcessor(accelerator)

    os.makedirs(output_folder, exist_ok=True)

    # Save model summary
    r, n = save_model_summary(output_folder, model)

    csv_logger = CSVLogger(output_folder, "logs_csv")
    tensorboard_logger = TensorBoardLogger(output_folder, "logs_tensorboard")

    resource_monitor = ResourceMonitor(
        interval=monitoring_interval,
        has_gpu=(accelerator_processor.accelerator == "gpu"),
        main_pid=os.getpid(),
        logging_level=logging.ERROR,
    )

    checkpoint_callback = ModelCheckpoint(
        output_folder,
        "model_lowest_valid_loss",
        save_last=True,
        monitor="validation_loss",
        mode="min",
        save_on_train_epoch_end=False,
        every_n_epochs=checkpoint_period
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "model_final_epoch"

    # write static information to a CSV file
    static_logfile_name = os.path.join(output_folder, "constants.csv")
    static_information_to_csv(
        static_logfile_name, accelerator_processor.to_torch(), n
    )

    if accelerator_processor.device is None:
        devices = "auto"
    else:
        devices = accelerator_processor.device

    with resource_monitor:

        trainer = Trainer(
            accelerator=accelerator_processor.accelerator,
            devices=devices,
            max_epochs=max_epoch,
            accumulate_grad_batches=batch_chunk_count,
            logger=[csv_logger, tensorboard_logger],
            check_val_every_n_epoch=1,
            callbacks=[LoggingCallback(resource_monitor), checkpoint_callback]
        )

        _ = trainer.fit(model, data_loader, valid_loader, ckpt_path=checkpoint)