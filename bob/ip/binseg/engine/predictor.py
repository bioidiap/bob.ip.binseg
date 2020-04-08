#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime

import PIL
import numpy
from tqdm import tqdm

import torch
import torchvision.transforms.functional as VF

import bob.io.base

import logging
logger = logging.getLogger(__name__)


def _save_hdf5(stem, prob, output_folder):
    """
    Saves prediction maps as image in the same format as the test image


    Parameters
    ----------
    stem : str
        the name of the file without extension on the original dataset

    prob : PIL.Image.Image
        Monochrome Image with prediction maps

    output_folder : str
        path where to store overlayed results

    """

    fullpath = os.path.join(output_folder, f"{stem}.hdf5")
    tqdm.write(f"Saving {fullpath}...")
    fulldir = os.path.dirname(fullpath)
    if not os.path.exists(fulldir):
        tqdm.write(f"Creating directory {fulldir}...")
        os.makedirs(fulldir, exist_ok=True)
    bob.io.base.save(prob.cpu().squeeze(0).numpy(), fullpath)


def _save_image(stem, extension, data, output_folder):
    """Saves a PIL image into a file

    Parameters
    ----------

    stem : str
        the name of the file without extension on the original dataset

    extension : str
        an extension for the file to be saved (e.g. ``.png``)

    data : PIL.Image.Image
        RGB image with the original image, preloaded

    output_folder : str
        path where to store results

    """

    fullpath = os.path.join(output_folder, stem + extension)
    tqdm.write(f"Saving {fullpath}...")
    fulldir = os.path.dirname(fullpath)
    if not os.path.exists(fulldir):
        tqdm.write(f"Creating directory {fulldir}...")
        os.makedirs(fulldir, exist_ok=True)
    data.save(fullpath)


def _save_overlayed_png(stem, image, prob, output_folder):
    """Overlays prediction predictions vessel tree with original test image


    Parameters
    ----------

    stem : str
        the name of the file without extension on the original dataset

    image : torch.Tensor
        Tensor with RGB input image

    prob : torch.Tensor
        Tensor with 1-D prediction map

    output_folder : str
        path where to store results

    """

    image = VF.to_pil_image(image)
    prob = VF.to_pil_image(prob.cpu().squeeze(0))

    # color and overlay
    prob_green = PIL.ImageOps.colorize(prob, (0, 0, 0), (0, 255, 0))
    overlayed = PIL.Image.blend(image, prob_green, 0.4)
    _save_image(stem, '.png', overlayed, output_folder)


def _save_transformed_png(stem, image, output_folder):
    """Saves a PNG copy of the transformed input image to a folder


    Parameters
    ----------

    stem : str
        the name of the file without extension on the original dataset

    image : torch.Tensor
        Tensor with RGB input image

    output_folder : str
        path where to store overlayed results

    """

    _save_image(stem, '.png', VF.to_pil_image(image), output_folder)


def run(model, data_loader, device, output_folder, overlayed_folder,
        transformed_input_folder):
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
        folder where to store output prediction maps (HDF5 files) and model
        summary

    overlayed_folder : str
        folder where to store output images (PNG files)

    transformed_input_folder : str
        folder where to store input images, transformed through the input
        pipeline (PNG files)

    """

    logger.info("Start prediction")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Device: {device}")

    if not os.path.exists(output_folder):
        logger.debug(f"Creating output directory '{output_folder}'...")
        # protect against concurrent access - exist_ok=True
        os.makedirs(output_folder, exist_ok=True)

    model.eval().to(device)
    # Sigmoid for predictions
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

            predictions = sigmoid(outputs)

            batch_time = time.perf_counter() - start_time
            times.append(batch_time)
            len_samples.append(len(images))

            for name, img, prob in zip(names, images, predictions):
                stem = os.path.splitext(name)[0]
                _save_hdf5(stem, prob, output_folder)
                if overlayed_folder is not None:
                    _save_overlayed_png(stem, img, prob, overlayed_folder)
                if transformed_input_folder is not None:
                    _save_transformed_png(stem, img, transformed_input_folder)

    logger.info("End prediction")

    # report operational summary
    total_time = datetime.timedelta(seconds=int(time.time() - start_total_time))
    logger.info(f"Total time: {total_time}")

    average_batch_time = numpy.mean(times)
    logger.info(f"Average batch time: {average_batch_time:g}s")

    average_image_time = numpy.sum(numpy.array(times) * len_samples) / float(sum(len_samples))
    logger.info(f"Average image time: {average_image_time:g}s")

    # Save model summary
    summary_path = os.path.join(output_folder, "model-info.txt")
    logger.info(f"Saving model summary at {summary_path}...")

    with open(summary_path, "w") as f: summary(model, f)
