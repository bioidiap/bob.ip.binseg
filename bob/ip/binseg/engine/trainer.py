#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import copy
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

from ..utils.resources import ResourceMonitor, cpu_constants, gpu_constants
from ..utils.summary import summary

logger = logging.getLogger(__name__)


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


def check_gpu(device):
    """
    Check the device type and the availability of GPU.

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
    """
    Save a little summary of the model in a txt file.

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
    with open(summary_path, "wt") as f:
        r, n = summary(model)
        logger.info(f"Model has {n} parameters...")
        f.write(r)
    return r, n


def static_information_to_csv(static_logfile_name, device, n):
    """
    Save the static information in a csv file.

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
        if device.type == "cuda":
            logdata += gpu_constants()
        logdata += (("model_size", n),)
        logwriter = csv.DictWriter(f, fieldnames=[k[0] for k in logdata])
        logwriter.writeheader()
        logwriter.writerow(dict(k for k in logdata))


def check_exist_logfile(logfile_name, arguments):
    """
    Check existance of logfile (trainlog.csv),
    If the logfile exist the and the epochs number are still 0, The logfile will be replaced.

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
    """
    Creation of the logfile fields that will appear in the logfile.

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
        "training_losses",
        "learning_rate",
    )

    if valid_loader is not None:
        logfile_fields += ("validation_losses",)
    if extra_valid_loaders:
        logfile_fields += ("extra_validation_losses",)
    logfile_fields += tuple(
        ResourceMonitor.monitored_keys(device.type == "cuda")
    )
    return logfile_fields


def train_epoch(
    loader,
    model,
    model_backup,
    optimizer,
    device,
    criterion,
    nbr_tasks,
    cur_task,
    epoch,
    last_epoch_model,
):
    """Trains the model for a single epoch (through all batches)

    Parameters
    ----------

    loader : :py:class:`torch.utils.data.DataLoader`
        To be used to train the model

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)

    model_backup : :py:class:`torch.nn.Module`
        copy of model

    optimizer : :py:mod:`torch.optim`

    device : :py:class:`torch.device`
        device to use

    criterion : :py:class:`torch.nn.modules.loss._Loss`

    nbr_tasks : int
        number of tasks

    cur_task : int
        index of the current task

    epoch : int
        current epoch

    last_epoch_model : dict
        Weights of the previous epoch

    Returns
    -------

    loss : float
        A floating-point value corresponding the weighted average of this
        epoch's loss

    """
    batch_losses = []
    samples_in_batch = []
    loss_tasks = [[] for i in range(nbr_tasks)]
    # progress bar only on interactive jobs
    for samples in tqdm(loader, desc="train", leave=False, disable=None):

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
        sigmoid = torch.nn.Sigmoid()
        outputs_prob = sigmoid(outputs)
        targets = torch.empty(outputs.shape, device=device)
        new_masks = torch.empty(outputs.shape, device=device)

        if epoch < nbr_tasks:
            for t in range(nbr_tasks):
                if t == cur_task:
                    targets[:, t, :, :] = ground_truths[:, 0, :, :]
                else:
                    targets[:, t, :, :] = outputs_prob[:, t, :, :]
                new_masks[:, t, :, :] = masks[:, 0, :, :]
        else:
            model_backup.load_state_dict(last_epoch_model)
            with torch.no_grad(), torch_evaluation(model_backup):
                outputs2 = model_backup(images)
                outputs2_prob = sigmoid(outputs2)
                for t in range(nbr_tasks):
                    if t == cur_task:
                        targets[:, t, :, :] = ground_truths[:, 0, :, :]
                    else:
                        targets[:, t, :, :] = outputs2_prob[:, t, :, :]
                    new_masks[:, t, :, :] = masks[:, 0, :, :]

        if nbr_tasks > 1:
            for t in range(nbr_tasks):
                loss_tasks[t].append(
                    criterion(
                        outputs[:, t, :, :],
                        targets[:, t, :, :],
                        new_masks[:, t, :, :],
                    )
                )

        else:
            loss_tasks[0].append(criterion(outputs, ground_truths, masks))
        # data forwarding on the existing network

        print(loss_tasks)
        loss = criterion(outputs, targets, new_masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.debug(f"batch loss: {loss.item()}")

        batch_losses.append(loss.item())
        samples_in_batch.append(len(samples))
    for nbr in range(nbr_tasks):
        loss_tasks[nbr] = (sum(loss_tasks[nbr]) / len(loss_tasks[nbr])).item()
    return numpy.average(batch_losses, weights=samples_in_batch), loss_tasks


def validate_epoch(
    loader, model, device, criterion, pbar_desc, nbr_tasks, cur_task
):
    """
    Processes input samples and returns loss (scalar)


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

    nbr_tasks : int
        number of tasks

    cur_task : int
        index of the current task

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
            if nbr_tasks == 1:
                loss = criterion(outputs, ground_truths, masks)
            else:
                loss = criterion(
                    outputs[:, cur_task, :, :],
                    ground_truths[:, 0, :, :],
                    masks[:, 0, :, :],
                )
            batch_losses.append(loss.item())
            samples_in_batch.append(len(samples))

    return numpy.average(batch_losses, weights=samples_in_batch)


def checkpointer_process(
    checkpointer,
    checkpoint_period,
    valid_losses,
    lowest_validation_loss,
    arguments,
    epoch,
    max_epoch,
):
    """
    Process the checkpointer, save the final model and keep track of the best model.

    Parameters
    ----------

    checkpointer : :py:class:`bob.ip.binseg.utils.checkpointer.Checkpointer`
        checkpointer implementation

    checkpoint_period : int
        save a checkpoint every ``n`` epochs.  If set to ``0`` (zero), then do
        not save intermediary checkpoints

    valid_losses : list
        Current epoch validation losses of all the tasks

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
    # Calculate the mean of the validation losses
    valid_loss = sum(valid_losses) / len(valid_losses)
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
    loss_tasks,
    valid_losses,
    extra_valid_losses,
    optimizer,
    logwriter,
    logfile,
    resource_data,
):
    """
    Write log info in trainlog.csv

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

    loss_tasks : list
        List of losses by tasks

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

    # if valid_loss is not None:
    #    logdata += (("validation_loss", f"{valid_loss:.6f}"),)

    if loss_tasks:
        entry = numpy.array_str(
            numpy.array(loss_tasks),
            max_line_width=sys.maxsize,
            precision=6,
        )
        logdata += (("training_losses", entry),)
    if valid_losses:
        entry = numpy.array_str(
            numpy.array(valid_losses),
            max_line_width=sys.maxsize,
            precision=6,
        )
        logdata += (("validation_losses", entry),)
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
    data_loaders,
    valid_loaders,
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
):
    """
    Fits an FCN model using supervised learning and save it to disk.

    This method supports periodic checkpointing and the output of a
    CSV-formatted log with the evolution of some figures during training.


    Parameters
    ----------

    model : :py:class:`torch.nn.Module`
        Network (e.g. driu, hed, unet)

    data_loaders : :py:class:`list` of :py:class:`torch.utils.data.DataLoader`
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

    monitoring_interval : int, float
        interval, in seconds (or fractions), through which we should monitor
        resources during training.

    """

    start_epoch = arguments["epoch"]
    max_epoch = arguments["max_epoch"]

    check_gpu(device)

    os.makedirs(output_folder, exist_ok=True)

    # Save model summary
    r, n = save_model_summary(output_folder, model)

    # write static information to a CSV file
    static_logfile_name = os.path.join(output_folder, "constants.csv")

    static_information_to_csv(static_logfile_name, device, n)

    # Log continous information to (another) file
    logfile_name = os.path.join(output_folder, "trainlog.csv")

    check_exist_logfile(logfile_name, arguments)

    logfile_fields = create_logfile_fields(
        valid_loaders, extra_valid_loaders, device
    )

    # the lowest validation loss obtained so far - this value is updated only
    # if a validation set is available
    lowest_validation_loss = sys.float_info.max

    last_epoch_model = model.state_dict()

    nbr_tasks = len(data_loaders)
    with open(logfile_name, "a+", newline="") as logfile:
        logwriter = csv.DictWriter(logfile, fieldnames=logfile_fields)

        if arguments["epoch"] == 0:
            logwriter.writeheader()

        model.train()  # set training mode

        model.to(device)  # set/cast parameters to device

        model_backup = copy.deepcopy(model)

        model_backup.to(device)

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
            cur_task = epoch % nbr_tasks
            data_loader = data_loaders[cur_task]
            with ResourceMonitor(
                interval=monitoring_interval,
                has_gpu=(device.type == "cuda"),
                main_pid=os.getpid(),
                logging_level=logging.ERROR,
            ) as resource_monitor:

                # Epoch time
                start_epoch_time = time.time()

                train_loss, loss_tasks = train_epoch(
                    data_loader,
                    model,
                    model_backup,
                    optimizer,
                    device,
                    criterion,
                    nbr_tasks,
                    cur_task,
                    epoch,
                    last_epoch_model,
                )

                scheduler.step()
                valid_losses = []
                for tsk, valid_loader in enumerate(valid_loaders):

                    valid_loss = (
                        validate_epoch(
                            valid_loader,
                            model,
                            device,
                            criterion,
                            "valid",
                            nbr_tasks,
                            tsk,
                        )
                        if valid_loader is not None
                        else None
                    )
                    valid_losses.append(valid_loss)

                extra_valid_losses = []
                for pos, extra_valid_loader in enumerate(extra_valid_loaders):
                    loss = validate_epoch(
                        extra_valid_loader,
                        model,
                        device,
                        criterion,
                        f"xval@{pos+1}",
                    )
                    extra_valid_losses.append(loss)

            lowest_validation_loss = checkpointer_process(
                checkpointer,
                checkpoint_period,
                valid_losses,
                lowest_validation_loss,
                arguments,
                epoch,
                max_epoch,
            )

            last_epoch_model = model.state_dict()

            # computes ETA (estimated time-of-arrival; end of training) taking
            # into consideration previous epoch performance
            epoch_time = time.time() - start_epoch_time
            eta_seconds = epoch_time * (max_epoch - epoch)
            current_time = time.time() - start_training_time

            write_log_info(
                epoch,
                current_time,
                eta_seconds,
                train_loss,
                loss_tasks,
                valid_losses,
                extra_valid_losses,
                optimizer,
                logwriter,
                logfile,
                resource_monitor.data,
            )

        total_training_time = time.time() - start_training_time
        logger.info(
            f"Total training time: {datetime.timedelta(seconds=total_training_time)} ({(total_training_time/max_epoch):.4f}s in average per epoch)"
        )
        epoch = epoch + 1
        arguments["epoch"] = epoch
