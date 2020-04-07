#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime

import numpy
import torch
from tqdm import tqdm

import bob.io.base

import logging
logger = logging.getLogger(__name__)


def save_hdf5(predictions, names, output_folder):
    """
    Saves probability maps as image in the same format as the test image


    Parameters
    ----------
    predictions : :py:class:`torch.Tensor`
        tensor with pixel-wise probabilities

    names : list
        list of file names

    output_folder : str
        output path

    """

    for j in range(predictions.size()[0]):

        img = predictions.cpu().data[j].squeeze(0).numpy()
        filename = "{}.hdf5".format(names[j].split(".")[0])
        fullpath = os.path.join(output_folder, filename)
        tqdm.write(f"Saving {fullpath}...")
        fulldir = os.path.dirname(fullpath)
        if not os.path.exists(fulldir):
            tqdm.write(f"Creating directory {fulldir}...")
            # protect against concurrent access - exist_ok=True
            os.makedirs(fulldir, exist_ok=True)
        bob.io.base.save(img, fullpath)


def run(model, data_loader, device, output_folder):
    """
    Runs inference on input data, outputs HDF5 files with predictions

    Parameters
    ---------
    model : :py:class:`torch.nn.Module`
        neural network model (e.g. driu, hed, unet)

    data_loader : py:class:`torch.torch.utils.data.DataLoader`

    device : str
        device to use ``cpu`` or ``cuda:0``

    output_folder : str
        folder where to store output images (HDF5 files)

    """

    logger.info("Start prediction")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Device: {device}")

    if not os.path.exists(output_folder):
        logger.debug(f"Creating output directory '{output_folder}'...")
        # protect against concurrent access - exist_ok=True
        os.makedirs(output_folder, exist_ok=True)

    model.eval().to(device)
    # Sigmoid for probabilities
    sigmoid = torch.nn.Sigmoid()

    # Setup timers
    start_total_time = time.time()
    times = []
    len_samples = []

    for samples in tqdm(
            data_loader, desc="batches", leave=False, disable=None,
            ):

        names = samples[0]
        images = samples[1].to(device)

        with torch.no_grad():

            start_time = time.perf_counter()
            outputs = model(images)

            # necessary check for HED architecture that uses several outputs
            # for loss calculation instead of just the last concatfuse block
            if isinstance(outputs, list):
                outputs = outputs[-1]

            probabilities = sigmoid(outputs)

            batch_time = time.perf_counter() - start_time
            times.append(batch_time)
            len_samples.append(len(images))

            save_hdf5(probabilities, names, output_folder)

    logger.info("End prediction")

    # report operational summary
    total_time = datetime.timedelta(seconds=int(time.time() - start_total_time))
    logger.info(f"Total time: {total_time}")

    average_batch_time = numpy.mean(times)
    logger.info(f"Average batch time: {average_batch_time:g}s\n")

    average_image_time = numpy.sum(numpy.array(times) * len_samples) / float(sum(len_samples))
    logger.info(f"Average image time: {average_image_time:g}s\n")
