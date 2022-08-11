#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
import os
import time

import h5py
import numpy
import torch
import torchvision.transforms.functional as VF

from tqdm import tqdm

from ..data.utils import overlayed_bbox_image

logger = logging.getLogger(__name__)


def _save_hdf5(stem, pred, output_folder):
    """
    Saves prediction maps as image in the same format as the test image


    Parameters
    ----------
    stem : str
        the name of the file without extension on the original dataset

    pred : list
        list of bounding box, label, and logit score for that label

    output_folder : str
        path where to store predictions

    """

    fullpath = os.path.join(output_folder, f"{stem}.hdf5")
    tqdm.write(f"Saving {fullpath}...")
    os.makedirs(os.path.dirname(fullpath), exist_ok=True)

    with h5py.File(fullpath, "w") as f:
        box = pred[0].cpu().squeeze(0).numpy()
        label = pred[1].cpu().squeeze(0).numpy()
        score = pred[2].cpu().squeeze(0).numpy()
        f.create_dataset("box", data=box)
        f.create_dataset("label", data=label)
        f.create_dataset("score", data=score)


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
    os.makedirs(os.path.dirname(fullpath), exist_ok=True)
    data.save(fullpath)


def _save_overlayed_png(stem, image, pred, output_folder):
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
    pred = pred[0].cpu().squeeze(0).numpy()
    _save_image(stem, ".png", overlayed_bbox_image(image, pred), output_folder)


def run(model, data_loader, name, device, output_folder, overlayed_folder):
    """
    Runs inference on input data, outputs HDF5 files with predictions

    Parameters
    ---------
    model : :py:class:`torch.nn.Module`
        neural network model (e.g. driu, hed, unet)

    data_loader : py:class:`torch.torch.utils.data.DataLoader`

    name : str
        the local name of this dataset (e.g. ``train``, or ``test``), to be
        used when saving measures files.

    device : :py:class:`torch.device`
        device to use

    output_folder : str
        folder where to store output prediction maps (HDF5 files) and model
        summary

    overlayed_folder : str
        folder where to store output images (PNG files)

    """

    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    logger.info(f"Device: {device}")

    model.eval()  # set evaluation mode
    model.to(device)  # set/cast parameters to device

    # Setup timers
    start_total_time = time.time()
    times = []
    len_samples = []

    output_folder = os.path.join(output_folder, name)
    overlayed_folder = (
        os.path.join(overlayed_folder, name)
        if overlayed_folder is not None
        else overlayed_folder
    )

    for samples in tqdm(data_loader, desc="batches", leave=False, disable=None):

        names = samples[0]
        images = list(
            image.to(device, non_blocking=torch.cuda.is_available())
            for image in samples[1]
        )

        with torch.no_grad():

            start_time = time.perf_counter()
            outputs = model(images)

            # necessary check for HED/Little W-Net architecture that use
            # several outputs for loss calculation instead of just the last one
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]

            boxes = outputs["boxes"]
            labels = outputs["labels"]
            scores = outputs["scores"]

            batch_time = time.perf_counter() - start_time
            times.append(batch_time)
            len_samples.append(len(images))

            for stem, img, box, label, score in zip(
                names, images, boxes, labels, scores
            ):

                pred = [box, label, score]

                _save_hdf5(stem, pred, output_folder)
                if overlayed_folder is not None:
                    _save_overlayed_png(stem, img, pred, overlayed_folder)

    # report operational summary
    total_time = datetime.timedelta(seconds=int(time.time() - start_total_time))
    logger.info(f"Total time: {total_time}")

    average_batch_time = numpy.mean(times)
    logger.info(f"Average batch time: {average_batch_time:g}s")

    average_image_time = numpy.sum(numpy.array(times) * len_samples) / float(
        sum(len_samples)
    )
    logger.info(f"Average image time: {average_image_time:g}s")
