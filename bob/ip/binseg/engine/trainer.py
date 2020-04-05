#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import time
import datetime
import torch
import pandas as pd
from tqdm import tqdm

from bob.ip.binseg.utils.metric import SmoothedValue
from bob.ip.binseg.utils.plot import loss_curve

import logging

logger = logging.getLogger(__name__)


def do_train(
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
):
    """
    Train models and save it to disk.

    This method supports periodic checkpointing and the output of a
    CSV-formatted log with the evolution of some figures during training.


    Parameters
    ----------

    model : :py:class:`torch.nn.Module`
        Network (e.g. DRIU, HED, UNet)

    data_loader : :py:class:`torch.utils.data.DataLoader`

    optimizer : :py:mod:`torch.optim`

    criterion : :py:class:`torch.nn.modules.loss._Loss`
        loss function

    scheduler : :py:mod:`torch.optim`
        learning rate scheduler

    checkpointer : :py:class:`bob.ip.binseg.utils.checkpointer.DetectronCheckpointer`
        checkpointer implementation

    checkpoint_period : int
        save a checkpoint every ``n`` epochs.  If set to ``0`` (zero), then do
        not save intermediary checkpoints

    device : str
        device to use ``'cpu'`` or ``cuda:0``

    arguments : dict
        start end end epochs

    output_folder : str
        output path
    """

    logger.info("Start training")
    start_epoch = arguments["epoch"]
    max_epoch = arguments["max_epoch"]

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
        "learning-rate",
        "gpu-memory-megabytes",
    )
    with open(logfile_name, "a+", newline="") as logfile:
        logwriter = csv.DictWriter(logfile, fieldnames=logfile_fields)

        if arguments["epoch"] == 0:
            logwriter.writeheader()

        model.train().to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        # Total training timer
        start_training_time = time.time()

        for epoch in range(start_epoch, max_epoch):
            scheduler.step()
            losses = SmoothedValue(len(data_loader))
            epoch = epoch + 1
            arguments["epoch"] = epoch

            # Epoch time
            start_epoch_time = time.time()

            # progress bar only on interactive jobs
            for samples in tqdm(
                data_loader, desc="batches", leave=False, disable=None,
            ):

                # data forwarding on the existing network
                images = samples[1].to(device)
                ground_truths = samples[2].to(device)
                masks = None
                if len(samples) == 4:
                    masks = samples[-1].to(device)

                outputs = model(images)

                # loss evaluation and learning (backward step)
                loss = criterion(outputs, ground_truths, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss)
                logger.debug("batch loss: {}".format(loss.item()))

            if checkpoint_period and (epoch % checkpoint_period == 0):
                checkpointer.save("model_{:03d}".format(epoch), **arguments)

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

        logger.info("End of training")
        total_training_time = time.time() - start_training_time
        logger.info(
            f"Total training time: {datetime.timedelta(seconds=total_training_time)} ({(total_training_time/max_epoch):.4f}s in average per epoch)"
        )

    # plots a version of the CSV trainlog into a PDF
    logdf = pd.read_csv(logfile_name, header=0, names=logfile_fields)
    fig = loss_curve(logdf, title="Loss Evolution")
    figurefile_name = os.path.join(output_folder, "trainlog.pdf")
    logger.info(f"Saving {figurefile_name}")
    fig.savefig(figurefile_name)
