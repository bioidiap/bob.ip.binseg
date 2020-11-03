#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import time
import shutil
import datetime
import contextlib
import distutils.version

import torch
from tqdm import tqdm

from ..utils.measure import SmoothedValue
from ..utils.summary import summary
from ..utils.resources import cpu_constants, gpu_constants, cpu_log, gpu_log

import logging

logger = logging.getLogger(__name__)

PYTORCH_GE_110 = distutils.version.StrictVersion(torch.__version__) >= "1.1.0"


@contextlib.contextmanager
def torch_evaluation(model):
    """Context manager to turn ON/OFF model evaluation

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


def run(
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
):
    """
    Fits an FCN model using supervised learning and save it to disk.

    This method supports periodic checkpointing and the output of a
    CSV-formatted log with the evolution of some figures during training.


    Parameters
    ----------

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)

    data_loader : :py:class:`torch.utils.data.DataLoader`
        To be used to train the model

    valid_loader : :py:class:`torch.utils.data.DataLoader`
        To be used to validate the model and enable automatic checkpointing.
        If set to ``None``, then do not validate it.

    optimizer : :py:mod:`torch.optim`

    criterion : :py:class:`torch.nn.modules.loss._Loss`
        loss function

    scheduler : :py:mod:`torch.optim`
        learning rate scheduler

    checkpointer : :py:class:`bob.ip.binseg.utils.checkpointer.Checkpointer`
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
    """

    start_epoch = arguments["epoch"]
    max_epoch = arguments["max_epoch"]

    if device.type == "cuda":
        # asserts we do have a GPU
        assert bool(
            gpu_constants()
        ), f"Device set to '{device}', but nvidia-smi is not installed"

    os.makedirs(output_folder, exist_ok=True)

    # Save model summary
    summary_path = os.path.join(output_folder, "model_summary.txt")
    logger.info(f"Saving model summary at {summary_path}...")
    with open(summary_path, "wt") as f:
        r, n = summary(model)
        logger.info(f"Model has {n} parameters...")
        f.write(r)

    # write static information to a CSV file
    static_logfile_name = os.path.join(output_folder, "constants.csv")
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

    # Log continous information to (another) file
    logfile_name = os.path.join(output_folder, "trainlog.csv")

    if arguments["epoch"] == 0 and os.path.exists(logfile_name):
        backup = logfile_name + "~"
        if os.path.exists(backup):
            os.unlink(backup)
        shutil.move(logfile_name, backup)

    logfile_fields = (
        "epoch",
        "total_time",
        "eta",
        "average_loss",
        "median_loss",
        "learning_rate",
    )
    if valid_loader is not None:
        logfile_fields += ("validation_average_loss", "validation_median_loss")
    logfile_fields += tuple([k[0] for k in cpu_log()])
    if device.type == "cuda":
        logfile_fields += tuple([k[0] for k in gpu_log()])

    # the lowest validation loss obtained so far - this value is updated only
    # if a validation set is available
    lowest_validation_loss = sys.float_info.max

    with open(logfile_name, "a+", newline="") as logfile:
        logwriter = csv.DictWriter(logfile, fieldnames=logfile_fields)

        if arguments["epoch"] == 0:
            logwriter.writeheader()

        model.train()  # set training mode

        model.to(device)  # set/cast parameters to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # Total training timer
        start_training_time = time.time()

        for epoch in tqdm(
            range(start_epoch, max_epoch),
            desc="epoch",
            leave=False,
            disable=None,
        ):
            if not PYTORCH_GE_110:
                scheduler.step()
            losses = SmoothedValue(len(data_loader))
            epoch = epoch + 1
            arguments["epoch"] = epoch

            # Epoch time
            start_epoch_time = time.time()

            # progress bar only on interactive jobs
            for samples in tqdm(
                data_loader, desc="batch", leave=False, disable=None
            ):

                # data forwarding on the existing network
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

                outputs = model(images)
                loss = criterion(outputs, ground_truths, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss)
                logger.debug(f"batch loss: {loss.item()}")

            if PYTORCH_GE_110:
                scheduler.step()

            # calculates the validation loss if necessary
            valid_losses = None
            if valid_loader is not None:

                with torch.no_grad(), torch_evaluation(model):

                    valid_losses = SmoothedValue(len(valid_loader))
                    for samples in tqdm(
                        valid_loader, desc="valid", leave=False, disable=None
                    ):
                        # data forwarding on the existing network
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

                        outputs = model(images)
                        loss = criterion(outputs, ground_truths, masks)
                        valid_losses.update(loss)

            if checkpoint_period and (epoch % checkpoint_period == 0):
                checkpointer.save(f"model_{epoch:03d}", **arguments)

            if (
                valid_losses is not None
                and valid_losses.avg < lowest_validation_loss
            ):
                lowest_validation_loss = valid_losses.avg
                logger.info(
                    f"Found new low on validation set:"
                    f" {lowest_validation_loss:.6f}"
                )
                checkpointer.save(f"model_lowest_valid_loss", **arguments)

            if epoch >= max_epoch:
                checkpointer.save("model_final", **arguments)

            # computes ETA (estimated time-of-arrival; end of training) taking
            # into consideration previous epoch performance
            epoch_time = time.time() - start_epoch_time
            eta_seconds = epoch_time * (max_epoch - epoch)
            current_time = time.time() - start_training_time

            logdata = (
                ("epoch", f"{epoch}"),
                (
                    "total_time",
                    f"{datetime.timedelta(seconds=int(current_time))}",
                ),
                ("eta", f"{datetime.timedelta(seconds=int(eta_seconds))}"),
                ("average_loss", f"{losses.avg:.6f}"),
                ("median_loss", f"{losses.median:.6f}"),
                ("learning_rate", f"{optimizer.param_groups[0]['lr']:.6f}"),
            )
            if valid_losses is not None:
                logdata += (
                    ("validation_average_loss", f"{valid_losses.avg:.6f}"),
                    ("validation_median_loss", f"{valid_losses.median:.6f}"),
                )
            logdata += cpu_log()
            if device.type == "cuda":
                logdata += gpu_log()

            logwriter.writerow(dict(k for k in logdata))
            logfile.flush()
            tqdm.write("|".join([f"{k}: {v}" for (k, v) in logdata[:4]]))

        total_training_time = time.time() - start_training_time
        logger.info(
            f"Total training time: {datetime.timedelta(seconds=total_training_time)} ({(total_training_time/max_epoch):.4f}s in average per epoch)"
        )
