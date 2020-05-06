#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import time
import datetime
import distutils.version

import numpy
import pandas
import torch
from tqdm import tqdm

from ..utils.measure import SmoothedValue
from ..utils.plot import loss_curve

import logging
logger = logging.getLogger(__name__)

PYTORCH_GE_110 = (distutils.version.StrictVersion(torch.__version__) >= "1.1.0")


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
    # TODO:
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
            l * unlabelled_input + (1 - l) * w_inputs[idx[: len(unlabelled_input)]]
        )
        unlabled_target_mixedup = (
            l * unlabled_target + (1 - l) * w_targets[idx[: len(unlabled_target)]]
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

    Parameters
    ----------

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)

    data_loader : :py:class:`torch.utils.data.DataLoader`

    optimizer : :py:mod:`torch.optim`

    criterion : :py:class:`torch.nn.modules.loss._Loss`
        loss function

    scheduler : :py:mod:`torch.optim`
        learning rate scheduler

    checkpointer : :py:class:`bob.ip.binseg.utils.checkpointer.DetectronCheckpointer`
        checkpointer

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

    if not os.path.exists(output_folder):
        logger.debug(f"Creating output directory '{output_folder}'...")
        os.makedirs(output_folder)

    # Log to file
    logfile_name = os.path.join(output_folder, "trainlog.csv")

    if arguments["epoch"] == 0 and os.path.exists(logfile_name):
        logger.info(f"Truncating {logfile_name} - training is restarting...")
        os.unlink(logfile_name)

    logfile_fields = (
        "epoch",
        "total-time",
        "eta",
        "average-loss",
        "median-loss",
        "median-labelled-loss",
        "median-unlabelled-loss",
        "learning-rate",
        "gpu-memory-megabytes",
    )
    with open(logfile_name, "a+", newline="") as logfile:
        logwriter = csv.DictWriter(logfile, fieldnames=logfile_fields)

        if arguments["epoch"] == 0:
            logwriter.writeheader()

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        model.train().to(device)
        # Total training timer
        start_training_time = time.time()

        for epoch in range(start_epoch, max_epoch):
            if not PYTORCH_GE_110: scheduler.step()
            losses = SmoothedValue(len(data_loader))
            labelled_loss = SmoothedValue(len(data_loader))
            unlabelled_loss = SmoothedValue(len(data_loader))
            epoch = epoch + 1
            arguments["epoch"] = epoch

            # Epoch time
            start_epoch_time = time.time()

            for samples in tqdm(data_loader, desc="batches", leave=False,
                    disable=None,):

                # data forwarding on the existing network

                # labelled
                images = samples[1].to(device)
                ground_truths = samples[2].to(device)
                unlabelled_images = samples[4].to(device)
                # labelled outputs
                outputs = model(images)
                unlabelled_outputs = model(unlabelled_images)
                # guessed unlabelled outputs
                unlabelled_ground_truths = guess_labels(unlabelled_images, model)
                # unlabelled_ground_truths = sharpen(unlabelled_ground_truths,0.5)
                # images, ground_truths, unlabelled_images, unlabelled_ground_truths = mix_up(0.75, images, ground_truths, unlabelled_images, unlabelled_ground_truths)

                # loss evaluation and learning (backward step)
                ramp_up_factor = square_rampup(epoch, rampup_length=rampup_length)

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

            if PYTORCH_GE_110: scheduler.step()

            if checkpoint_period and (epoch % checkpoint_period == 0):
                checkpointer.save(f"model_{epoch:03d}", **arguments)

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
                    "total-time",
                    f"{datetime.timedelta(seconds=int(current_time))}",
                ),
                ("eta", f"{datetime.timedelta(seconds=int(eta_seconds))}"),
                ("average-loss", f"{losses.avg:.6f}"),
                ("median-loss", f"{losses.median:.6f}"),
                ("median-labelled-loss", f"{labelled_loss.median:.6f}"),
                ("median-unlabelled-loss", f"{unlabelled_loss.median:.6f}"),
                ("learning-rate", f"{optimizer.param_groups[0]['lr']:.6f}"),
                (
                    "gpu-memory-megabytes",
                    f"{torch.cuda.max_memory_allocated()/(1024.0*1024.0)}"
                    if torch.cuda.is_available()
                    else "0.0",
                ),
            )
            logwriter.writerow(dict(k for k in logdata))
            logger.info("|".join([f"{k}: {v}" for (k, v) in logdata]))

        total_training_time = time.time() - start_training_time
        logger.info(
            f"Total training time: {datetime.timedelta(seconds=total_training_time)} ({(total_training_time/max_epoch):.4f}s in average per epoch)"
        )

    # plots a version of the CSV trainlog into a PDF
    logdf = pandas.read_csv(logfile_name, header=0, names=logfile_fields)
    fig = loss_curve(logdf)
    figurefile_name = os.path.join(output_folder, "trainlog.pdf")
    logger.info(f"Saving {figurefile_name}")
    fig.savefig(figurefile_name)
