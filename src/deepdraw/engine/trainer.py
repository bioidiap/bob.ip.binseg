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


@contextlib.contextmanager
def torch_evaluation(model):
    """Context manager to turn ON/OFF model evaluation.

    This context manager will turn evaluation mode ON on entry and turn it OFF
    when exiting the ``with`` statement block.


    Parameters
    ----------

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)


    Yields
    ------

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)
    """

    model.eval()
    yield model
    model.train()


def check_gpu(device):
    """Check the device type and the availability of GPU.

    Parameters
    ----------

    device : :py:class:`torch.device`
        device to use
    """
    if device.type == "cuda":
        # asserts we do have a GPU
        assert bool(
            gpu_constants()
        ), f"Device set to '{device}', but nvidia-smi is not installed"


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


def check_exist_logfile(logfile_name, arguments):
    """Check existance of logfile (trainlog.csv), If the logfile exist the and
    the epochs number are still 0, The logfile will be replaced.

    Parameters
    ----------

    logfile_name : str
        The logfile_name which is a join between the output_folder and trainlog.csv

    arguments : dict
        start and end epochs
    """
    if arguments["epoch"] == 0 and os.path.exists(logfile_name):
        backup = logfile_name + "~"
        if os.path.exists(backup):
            os.unlink(backup)
        shutil.move(logfile_name, backup)


def create_logfile_fields(valid_loader, extra_valid_loaders, device):
    """Creation of the logfile fields that will appear in the logfile.

    Parameters
    ----------

    valid_loader : :py:class:`torch.utils.data.DataLoader`
        To be used to validate the model and enable automatic checkpointing.
        If set to ``None``, then do not validate it.

    extra_valid_loaders : :py:class:`list` of :py:class:`torch.utils.data.DataLoader`
        To be used to validate the model, however **does not affect** automatic
        checkpointing. If set to ``None``, or empty, then does not log anything
        else.  Otherwise, an extra column with the loss of every dataset in
        this list is kept on the final training log.

    device : :py:class:`torch.device`
        device to use

    Returns
    -------

    logfile_fields: tuple
        The fields that will appear in trainlog.csv
    """
    logfile_fields = (
        "epoch",
        "total_time",
        "eta",
        "loss",
        "learning_rate",
    )
    if valid_loader is not None:
        logfile_fields += ("validation_loss",)
    if extra_valid_loaders:
        logfile_fields += ("extra_validation_losses",)
    logfile_fields += tuple(
        ResourceMonitor.monitored_keys(device.type == "cuda")
    )
    return logfile_fields


def train_epoch(loader, model, optimizer, device, criterion, batch_chunk_count):
    """Trains the model for a single epoch (through all batches)

    Parameters
    ----------

    loader : :py:class:`torch.utils.data.DataLoader`
        To be used to train the model

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)

    optimizer : :py:mod:`torch.optim`

    device : :py:class:`torch.device`
        device to use

    criterion : :py:class:`torch.nn.modules.loss._Loss`

    batch_chunk_count: int
        If this number is different than 1, then each batch will be divided in
        this number of chunks.  Gradients will be accumulated to perform each
        mini-batch.   This is particularly interesting when one has limited RAM
        on the GPU, but would like to keep training with larger batches.  One
        exchanges for longer processing times in this case.  To better understand
        gradient accumulation, read
        https://stackoverflow.com/questions/62067400/understanding-accumulated-gradients-in-pytorch.


    Returns
    -------

    loss : float
        A floating-point value corresponding the weighted average of this
        epoch's loss
    """

    losses_in_epoch = []
    samples_in_epoch = []
    losses_in_batch = []
    samples_in_batch = []

    # progress bar only on interactive jobs
    for idx, samples in enumerate(
        tqdm(loader, desc="train", leave=False, disable=None)
    ):
        images = samples[1].to(
            device=device, non_blocking=torch.cuda.is_available()
        )
        ground_truths = samples[2].to(
            device=device, non_blocking=torch.cuda.is_available()
        )
        masks = (
            torch.ones_like(ground_truths)
            if len(samples) < 4
            else samples[3].to(
                device=device, non_blocking=torch.cuda.is_available()
            )
        )

        # Forward pass on the network
        outputs = model(images)
        loss = criterion(outputs, ground_truths, masks)

        losses_in_batch.append(loss.item())
        samples_in_batch.append(len(samples))

        # Normalize loss to account for batch accumulation
        loss = loss / batch_chunk_count

        # Accumulate gradients - does not update weights just yet...
        loss.backward()

        # Weight update on the network
        if ((idx + 1) % batch_chunk_count == 0) or (idx + 1 == len(loader)):
            # Advances optimizer to the "next" state and applies weight update
            # over the whole model
            optimizer.step()

            # Zeroes gradients for the next batch
            optimizer.zero_grad()

            # Normalize loss for current batch
            batch_loss = numpy.average(
                losses_in_batch, weights=samples_in_batch
            )
            losses_in_epoch.append(batch_loss.item())
            samples_in_epoch.append(len(samples))

            losses_in_batch.clear()
            samples_in_batch.clear()
            logger.debug(f"batch loss: {batch_loss.item()}")

    return numpy.average(losses_in_epoch, weights=samples_in_epoch)


def validate_epoch(loader, model, device, criterion, pbar_desc):
    """Processes input samples and returns loss (scalar)

    Parameters
    ----------

    loader : :py:class:`torch.utils.data.DataLoader`
        To be used to validate the model

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)

    optimizer : :py:mod:`torch.optim`

    device : :py:class:`torch.device`
        device to use

    criterion : :py:class:`torch.nn.modules.loss._Loss`
        loss function

    pbar_desc : str
        A string for the progress bar descriptor


    Returns
    -------

    loss : float
        A floating-point value corresponding the weighted average of this
        epoch's loss
    """

    batch_losses = []
    samples_in_batch = []

    with torch.no_grad(), torch_evaluation(model):
        for samples in tqdm(loader, desc=pbar_desc, leave=False, disable=None):
            images = samples[1].to(
                device=device,
                non_blocking=torch.cuda.is_available(),
            )
            ground_truths = samples[2].to(
                device=device,
                non_blocking=torch.cuda.is_available(),
            )
            masks = (
                torch.ones_like(ground_truths)
                if len(samples) < 4
                else samples[3].to(
                    device=device,
                    non_blocking=torch.cuda.is_available(),
                )
            )

            # data forwarding on the existing network
            outputs = model(images)
            loss = criterion(outputs, ground_truths, masks)

            batch_losses.append(loss.item())
            samples_in_batch.append(len(samples))

    return numpy.average(batch_losses, weights=samples_in_batch)


def checkpointer_process(
    checkpointer,
    checkpoint_period,
    valid_loss,
    lowest_validation_loss,
    arguments,
    epoch,
    max_epoch,
):
    """Process the checkpointer, save the final model and keep track of the
    best model.

    Parameters
    ----------

    checkpointer : :py:class:`deepdraw.utils.checkpointer.Checkpointer`
        checkpointer implementation

    checkpoint_period : int
        save a checkpoint every ``n`` epochs.  If set to ``0`` (zero), then do
        not save intermediary checkpoints

    valid_loss : float
        Current epoch validation loss

    lowest_validation_loss : float
        Keeps track of the best (lowest) validation loss

    arguments : dict
        start and end epochs

    max_epoch : int
        end_potch

    Returns
    -------

    lowest_validation_loss : float
        The lowest validation loss currently observed
    """
    if checkpoint_period and (epoch % checkpoint_period == 0):
        checkpointer.save("model_periodic_save", **arguments)

    if valid_loss is not None and valid_loss < lowest_validation_loss:
        lowest_validation_loss = valid_loss
        logger.info(
            f"Found new low on validation set:" f" {lowest_validation_loss:.6f}"
        )
        checkpointer.save("model_lowest_valid_loss", **arguments)

    if epoch >= max_epoch:
        checkpointer.save("model_final_epoch", **arguments)

    return lowest_validation_loss


def write_log_info(
    epoch,
    current_time,
    eta_seconds,
    loss,
    valid_loss,
    extra_valid_losses,
    optimizer,
    logwriter,
    logfile,
    resource_data,
):
    """Write log info in trainlog.csv.

    Parameters
    ----------

    epoch : int
        Current epoch

    current_time : float
        Current training time

    eta_seconds : float
        estimated time-of-arrival taking into consideration previous epoch performance

    loss : float
        Current epoch's training loss

    valid_loss : :py:class:`float`, None
        Current epoch's validation loss

    extra_valid_losses : :py:class:`list` of :py:class:`float`
        Validation losses from other validation datasets being currently
        tracked

    optimizer : :py:mod:`torch.optim`

    logwriter : csv.DictWriter
        Dictionary writer that give the ability to write on the trainlog.csv

    logfile : io.TextIOWrapper

    resource_data : tuple
        Monitored resources at the machine (CPU and GPU)
    """

    logdata = (
        ("epoch", f"{epoch}"),
        (
            "total_time",
            f"{datetime.timedelta(seconds=int(current_time))}",
        ),
        ("eta", f"{datetime.timedelta(seconds=int(eta_seconds))}"),
        ("loss", f"{loss:.6f}"),
        ("learning_rate", f"{optimizer.param_groups[0]['lr']:.6f}"),
    )

    if valid_loss is not None:
        logdata += (("validation_loss", f"{valid_loss:.6f}"),)

    if extra_valid_losses:
        entry = numpy.array_str(
            numpy.array(extra_valid_losses),
            max_line_width=sys.maxsize,
            precision=6,
        )
        logdata += (("extra_validation_losses", entry),)

    logdata += resource_data

    logwriter.writerow(dict(k for k in logdata))
    logfile.flush()
    tqdm.write("|".join([f"{k}: {v}" for (k, v) in logdata[:4]]))


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