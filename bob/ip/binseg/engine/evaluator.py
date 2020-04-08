#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines functionality for the evaluation of predictions"""

import os

import PIL
import numpy
import pandas
from tqdm import tqdm

import torch
import torchvision.transforms.functional as VF

import bob.io.base

from ..utils.metric import base_metrics
from ..utils.plot import precision_recall_f1iso_confintval
from ..utils.summary import summary

import logging
logger = logging.getLogger(__name__)


def _posneg(pred, gt, threshold):
    """Calculates true and false positives and negatives"""

    gt = gt.byte()  # byte tensor

    # threshold
    binary_pred = torch.gt(pred, threshold).byte()

    # equals and not-equals
    equals = torch.eq(binary_pred, gt).type(torch.uint8)  # tensor
    notequals = torch.ne(binary_pred, gt).type(torch.uint8)  # tensor

    # true positives
    tp_tensor = gt * binary_pred

    # false positives
    fp_tensor = torch.eq((binary_pred + tp_tensor), 1)

    # true negatives
    tn_tensor = equals - tp_tensor

    # false negatives
    fn_tensor = notequals - fp_tensor.type(torch.uint8)

    return tp_tensor, fp_tensor, tn_tensor, fn_tensor


def _sample_metrics(pred, gt):
    """
    Calculates metrics on one single sample and saves it to disk


    Parameters
    ----------

    pred : torch.Tensor
        pixel-wise predictions

    gt : torch.Tensor
        ground-truth (annotations)


    Returns
    -------

    metrics : pandas.DataFrame

        A pandas dataframe with the following columns:

        * threshold: float
        * precision: float
        * recall: float
        * specificity: float
        * accuracy: float
        * jaccard: float
        * f1_score: float

    """

    step_size = 0.01
    data = []

    for threshold in numpy.arange(0.0, 1.0, step_size):

        tp_tensor, fp_tensor, tn_tensor, fn_tensor = _posneg(pred, gt, threshold)

        # calc metrics from scalars
        tp_count = torch.sum(tp_tensor).item()
        fp_count = torch.sum(fp_tensor).item()
        tn_count = torch.sum(tn_tensor).item()
        fn_count = torch.sum(fn_tensor).item()
        precision, recall, specificity, accuracy, jaccard, f1_score = \
                base_metrics(tp_count, fp_count, tn_count, fn_count)

        data.append([threshold, precision, recall, specificity,
            accuracy, jaccard, f1_score])

    return pandas.DataFrame(data, columns=(
        "threshold",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "jaccard",
        "f1_score",
        ))


def _sample_analysis(
        img,
        pred,
        gt,
        threshold,
        tp_color=(0, 255, 0),  # (128,128,128) Gray
        fp_color=(0, 0, 255),  # (70, 240, 240) Cyan
        fn_color=(255, 0, 0),  # (245, 130, 48) Orange
        overlay=True,
        ):
    """Visualizes true positives, false positives and false negatives


    Parameters
    ----------

    img : torch.Tensor
        original image

    pred : torch.Tensor
        pixel-wise predictions

    gt : torch.Tensor
        ground-truth (annotations)

    threshold : float
        The threshold to be used while analyzing this image's probability map

    tp_color : tuple
        RGB value for true positives

    fp_color : tuple
        RGB value for false positives

    fn_color : tuple
        RGB value for false negatives

    overlay : :py:class:`bool`, Optional
        If set to ``True`` (which is the default), then overlay annotations on
        top of the image.  Otherwise, represent data on a black canvas.


    Returns
    -------

    figure : PIL.Image.Image

        A PIL image that contains the overlayed analysis of true-positives
        (TP), false-positives (FP) and false negatives (FN).

    """

    tp_tensor, fp_tensor, tn_tensor, fn_tensor = _posneg(pred, gt, threshold)

    # change to PIL representation
    tp_pil = VF.to_pil_image(tp_tensor.float())
    tp_pil_colored = PIL.ImageOps.colorize(tp_pil, (0, 0, 0), tp_color)

    fp_pil = VF.to_pil_image(fp_tensor.float())
    fp_pil_colored = PIL.ImageOps.colorize(fp_pil, (0, 0, 0), fp_color)

    fn_pil = VF.to_pil_image(fn_tensor.float())
    fn_pil_colored = PIL.ImageOps.colorize(fn_pil, (0, 0, 0), fn_color)

    tp_pil_colored.paste(fp_pil_colored, mask=fp_pil)
    tp_pil_colored.paste(fn_pil_colored, mask=fn_pil)

    if overlay:
        img = VF.to_pil_image(img)  # PIL Image
        tp_pil_colored = PIL.Image.blend(img, tp_pil_colored, 0.4)

    return tp_pil_colored


def run(data_loader, predictions_folder, output_folder, overlayed_folder=None,
        overlay_threshold=None):
    """
    Runs inference and calculates metrics


    Parameters
    ---------

    data_loader : py:class:`torch.torch.utils.data.DataLoader`
        an iterable over the transformed input dataset, containing ground-truth

    predictions_folder : str
        folder where predictions for the dataset images has been previously
        stored

    output_folder : str
        folder where to store results

    overlayed_folder : :py:class:`str`, Optional
        if not ``None``, then it should be the name of a folder where to store
        overlayed versions of the images and ground-truths

    overlay_threshold : :py:class:`float`, Optional
        if ``overlayed_folder``, then this should be threshold (floating point)
        to apply to prediction maps to decide on positives and negatives for
        overlaying analysis (graphical output).  This number should come from
        the training set or a separate validation set.  Using a test set value
        may bias your analysis.

    """

    logger.info("Start evaluation")
    logger.info(f"Output folder: {output_folder}")

    if not os.path.exists(output_folder):
        logger.info(f"Creating {output_folder}...")
        os.makedirs(output_folder, exist_ok=True)

    # Collect overall metrics
    data = {}

    for sample in tqdm(data_loader):
        name = sample[0]
        stem = os.path.splitext(name)[0]
        image = sample[1].to("cpu")
        gt = sample[2].to("cpu")
        pred_fullpath = os.path.join(predictions_folder, stem + ".hdf5")
        pred = bob.io.base.load(pred_fullpath).astype("float32")
        pred = torch.from_numpy(pred)
        if stem in data:
            raise RuntimeError(f"{stem} entry already exists in data. "
                    f"Cannot overwrite.")
        data[stem] = _sample_metrics(pred, gt)

        if overlayed_folder is not None:
            overlay_image = _sample_analysis(image, pred, gt,
                    threshold=overlay_threshold, overlay=True)
            fullpath = os.path.join(overlayed_folder, f"{stem}.png")
            tqdm.write(f"Saving {fullpath}...")
            fulldir = os.path.dirname(fullpath)
            if not os.path.exists(fulldir):
                tqdm.write(f"Creating directory {fulldir}...")
                os.makedirs(fulldir, exist_ok=True)
            overlay_image.save(fullpath)

    # Merges all dataframes together
    df_metrics = pandas.concat(data.values())

    # Report and Averages
    avg_metrics = df_metrics.groupby("threshold").mean()
    std_metrics = df_metrics.groupby("threshold").std()

    # Uncomment below for F1-score calculation based on average precision and
    # metrics instead of F1-scores of individual images. This method is in line
    # with Maninis et. al. (2016)
    #
    # avg_metrics["f1_score"] = \
    #         (2* avg_metrics["precision"]*avg_metrics["recall"])/ \
    #         (avg_metrics["precision"]+avg_metrics["recall"])

    avg_metrics["std_pr"] = std_metrics["precision"]
    avg_metrics["pr_upper"] = avg_metrics["precision"] + avg_metrics["std_pr"]
    avg_metrics["pr_lower"] = avg_metrics["precision"] - avg_metrics["std_pr"]
    avg_metrics["std_re"] = std_metrics["recall"]
    avg_metrics["re_upper"] = avg_metrics["recall"] + avg_metrics["std_re"]
    avg_metrics["re_lower"] = avg_metrics["recall"] - avg_metrics["std_re"]
    avg_metrics["std_f1"] = std_metrics["f1_score"]

    metrics_path = os.path.join(output_folder, "metrics.csv")
    logger.info(f"Saving averages over all input images at {metrics_path}...")
    avg_metrics.to_csv(metrics_path)

    maxf1 = avg_metrics["f1_score"].max()
    optimal_f1_threshold = avg_metrics["f1_score"].idxmax()

    logger.info(f"Highest F1-score of {maxf1:.5f}, achieved at "
            f"threshold {optimal_f1_threshold:.2f}")

    # Plotting
    np_avg_metrics = avg_metrics.to_numpy().T
    figure_path = os.path.join(output_folder, "precision-recall.pdf")
    logger.info(f"Saving overall precision-recall plot at {figure_path}...")
    fig = precision_recall_f1iso_confintval(
        [np_avg_metrics[0]],
        [np_avg_metrics[1]],
        [np_avg_metrics[7]],
        [np_avg_metrics[8]],
        [np_avg_metrics[10]],
        [np_avg_metrics[11]],
        ["data"],
    )
    fig.savefig(figure_path)
