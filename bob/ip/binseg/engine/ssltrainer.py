#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import time
import shutil
import datetime
import distutils.version

import numpy
import torch
from tqdm import tqdm

from ..utils.measure import SmoothedValue
from ..utils.summary import summary
from ..utils.resources import cpu_constants, gpu_constants, cpu_log, gpu_log
from .trainer import PYTORCH_GE_110, torch_evaluation

import logging

logger = logging.getLogger(__name__)


def sharpen(x, T):
    temp = x ** (1 / T)
    return temp / temp.sum(dim=1, keepdim=True)


def mix_up(alpha, input, target, unlabelled_input, unlabled_target):
    """Applies mix up as described in [MIXMATCH_19].

    Parameters
    ----------
    alpha : float

    input : :py:class:`torch.Tensor`

    target : :py:class:`torch.Tensor`

    unlabelled_input : :py:class:`torch.Tensor`

    unlabled_target : :py:class:`torch.Tensor`


    Returns
    -------

    list

    """

    with torch.no_grad():
        l = numpy.random.beta(alpha, alpha)  # Eq (8)
        l = max(l, 1 - l)  # Eq (9)
        # Shuffle and concat. Alg. 1 Line: 12
        w_inputs = torch.cat([input, unlabelled_input], 0)
        w_targets = torch.cat([target, unlabled_target], 0)
        idx = torch.randperm(w_inputs.size(0))  # get random index

        # Apply MixUp to labelled data and entries from W. Alg. 1 Line: 13
        input_mixedup = l * input + (1 - l) * w_inputs[idx[len(input) :]]
        target_mixedup = l * target + (1 - l) * w_targets[idx[len(target) :]]

        # Apply MixUp to unlabelled data and entries from W. Alg. 1 Line: 14
        unlabelled_input_mixedup = (
            l * unlabelled_input
            + (1 - l) * w_inputs[idx[: len(unlabelled_input)]]
        )
        unlabled_target_mixedup = (
            l * unlabled_target
            + (1 - l) * w_targets[idx[: len(unlabled_target)]]
        )
        return (
            input_mixedup,
            target_mixedup,
            unlabelled_input_mixedup,
            unlabled_target_mixedup,
        )


def square_rampup(current, rampup_length=16):
    """slowly ramp-up ``lambda_u``

    Parameters
    ----------

    current : int
        current epoch

    rampup_length : :obj:`int`, optional
        how long to ramp up, by default 16

    Returns
    -------

    factor : float
        ramp up factor
    """

    if rampup_length == 0:
        return 1.0
    else:
        current = numpy.clip((current / float(rampup_length)) ** 2, 0.0, 1.0)
    return float(current)


def linear_rampup(current, rampup_length=16):
    """slowly ramp-up ``lambda_u``

    Parameters
    ----------
    current : int
        current epoch

    rampup_length : :obj:`int`, optional
        how long to ramp up, by default 16

    Returns
    -------

    factor: float
        ramp up factor

    """
    if rampup_length == 0:
        return 1.0
    else:
        current = numpy.clip(current / rampup_length, 0.0, 1.0)
    return float(current)


def guess_labels(unlabelled_images, model):
    """
    Calculate the average predictions by 2 augmentations: horizontal and vertical flips

    Parameters
    ----------

    unlabelled_images : :py:class:`torch.Tensor`
        ``[n,c,h,w]``

    target : :py:class:`torch.Tensor`

    Returns
    -------

    shape : :py:class:`torch.Tensor`
        ``[n,c,h,w]``

    """
    with torch.no_grad():
        guess1 = torch.sigmoid(model(unlabelled_images)).unsqueeze(0)
        # Horizontal flip and unsqueeze to work with batches (increase flip dimension by 1)
        hflip = torch.sigmoid(model(unlabelled_images.flip(2))).unsqueeze(0)
        guess2 = hflip.flip(3)
        # Vertical flip and unsqueeze to work with batches (increase flip dimension by 1)
        vflip = torch.sigmoid(model(unlabelled_images.flip(3))).unsqueeze(0)
        guess3 = vflip.flip(4)
        # Concat
        concat = torch.cat([guess1, guess2, guess3], 0)
        avg_guess = torch.mean(concat, 0)
        return avg_guess


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
    rampup_length,
):
    """
    Fits an FCN model using semi-supervised learning and saves it to disk.


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

    device : str
        device to use ``'cpu'`` or ``cuda:0``

    arguments : dict
        start and end epochs

    output_folder : str
        output path

    rampup_length : int
        rampup epochs

    """

    start_epoch = arguments["epoch"]
    max_epoch = arguments["max_epoch"]

    if device != "cpu":
        # asserts we do have a GPU
        assert bool(gpu_constants()), (
            f"Device set to '{device}', but cannot "
            f"find a GPU (maybe nvidia-smi is not installed?)"
        )

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
        if device != "cpu":
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
        "labelled_median_loss",
        "unlabelled_median_loss",
        "learning_rate",
    )
    if valid_loader is not None:
        logfile_fields += ("validation_average_loss", "validation_median_loss")
    logfile_fields += tuple([k[0] for k in cpu_log()])
    if device != "cpu":
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
            labelled_loss = SmoothedValue(len(data_loader))
            unlabelled_loss = SmoothedValue(len(data_loader))
            epoch = epoch + 1
            arguments["epoch"] = epoch

            # Epoch time
            start_epoch_time = time.time()

            # progress bar only on interactive jobs
            for samples in tqdm(
                data_loader, desc="batch", leave=False, disable=None
            ):

                # data forwarding on the existing network

                # labelled
                images = samples[1].to(
                    device=device, non_blocking=torch.cuda.is_available()
                )
                ground_truths = samples[2].to(
                    device=device, non_blocking=torch.cuda.is_available()
                )
                unlabelled_images = samples[4].to(
                    device=device, non_blocking=torch.cuda.is_available()
                )
                # labelled outputs
                outputs = model(images)
                unlabelled_outputs = model(unlabelled_images)
                # guessed unlabelled outputs
                unlabelled_ground_truths = guess_labels(
                    unlabelled_images, model
                )

                # loss evaluation and learning (backward step)
                ramp_up_factor = square_rampup(
                    epoch, rampup_length=rampup_length
                )

                # note: no support for masks...
                loss, ll, ul = criterion(
                    outputs,
                    ground_truths,
                    unlabelled_outputs,
                    unlabelled_ground_truths,
                    ramp_up_factor,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss)
                labelled_loss.update(ll)
                unlabelled_loss.update(ul)
                logger.debug(f"batch loss: {loss.item()}")

            if PYTORCH_GE_110:
                scheduler.step()

            # calculates the validation loss if necessary
            # note: validation does not comprise "unlabelled" losses
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
                ("labelled_median_loss", f"{labelled_loss.median:.6f}"),
                ("unlabelled_median_loss", f"{unlabelled_loss.median:.6f}"),
                ("learning_rate", f"{optimizer.param_groups[0]['lr']:.6f}"),
            )
            if valid_losses is not None:
                logdata += (
                    ("validation_average_loss", f"{valid_losses.avg:.6f}"),
                    ("validation_median_loss", f"{valid_losses.median:.6f}"),
                )
            logdata += cpu_log()
            if device != "cpu":
                logdata += gpu_log()

            if device != "cpu":
                logdata += gpu_log()

            logwriter.writerow(dict(k for k in logdata))
            logfile.flush()
            tqdm.write("|".join([f"{k}: {v}" for (k, v) in logdata[:4]]))

        total_training_time = time.time() - start_training_time
        logger.info(
            f"Total training time: {datetime.timedelta(seconds=total_training_time)} ({(total_training_time/max_epoch):.4f}s in average per epoch)"
        )
