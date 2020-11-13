#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines functionality for the evaluation of predictions"""

import os

import PIL
import numpy
import pandas
from tqdm import tqdm

import torch
import torch.nn.functional
import torchvision.transforms.functional as VF

import h5py

from ..utils.measure import base_measures, bayesian_measures

import logging

logger = logging.getLogger(__name__)


def _posneg(pred, gt, threshold):
    """Calculates true and false positives and negatives


    Parameters
    ----------

    pred : torch.Tensor
        pixel-wise predictions

    gt : torch.Tensor
        ground-truth (annotations)

    threshold : float
        a particular threshold in which to calculate the performance
        measures


    Returns
    -------

    tp_tensor : torch.Tensor
        boolean tensor with true positives, considering all observations

    fp_tensor : torch.Tensor
        boolean tensor with false positives, considering all observations

    tn_tensor : torch.Tensor
        boolean tensor with true negatives, considering all observations

    fn_tensor : torch.Tensor
        boolean tensor with false negatives, considering all observations

    """

    gt = gt.byte()  # byte tensor

    # threshold
    binary_pred = torch.gt(pred, threshold).byte()

    # equals and not-equals
    equals = torch.eq(binary_pred, gt).type(torch.uint8)  # tensor
    notequals = torch.ne(binary_pred, gt).type(torch.uint8)  # tensor

    # true positives
    tp_tensor = gt * binary_pred

    # false positives
    fp_tensor = torch.eq((binary_pred + tp_tensor), 1).byte()

    # true negatives
    tn_tensor = equals - tp_tensor

    # false negatives
    fn_tensor = notequals - fp_tensor

    return tp_tensor, fp_tensor, tn_tensor, fn_tensor


def sample_measures_for_threshold(pred, gt, mask, threshold):
    """
    Calculates counts on one single sample, for a specific threshold


    Parameters
    ----------

    pred : torch.Tensor
        pixel-wise predictions

    gt : torch.Tensor
        ground-truth (annotations)

    mask : torch.Tensor
        region mask (used only if available).  May be set to ``None``.

    threshold : float
        a particular threshold in which to calculate the performance
        measures


    Returns
    -------

    tp : int

    fp : int

    tn : int

    fn : int

    """

    tp_tensor, fp_tensor, tn_tensor, fn_tensor = _posneg(pred, gt, threshold)

    # if a mask is provided, consider only TP/FP/TN/FN **within** the region of
    # interest defined by the mask
    if mask is not None:
        antimask = torch.le(mask, 0.5)
        tp_tensor[antimask] = 0
        fp_tensor[antimask] = 0
        tn_tensor[antimask] = 0
        fn_tensor[antimask] = 0

    # calc measures from scalars
    tp_count = torch.sum(tp_tensor).item()
    fp_count = torch.sum(fp_tensor).item()
    tn_count = torch.sum(tn_tensor).item()
    fn_count = torch.sum(fn_tensor).item()

    return tp_count, fp_count, tn_count, fn_count


def _sample_measures(pred, gt, mask, steps):
    """
    Calculates measures on one single sample


    Parameters
    ----------

    pred : torch.Tensor
        pixel-wise predictions

    gt : torch.Tensor
        ground-truth (annotations)

    mask : torch.Tensor
        region mask (used only if available).  May be set to ``None``.

    steps : int
        number of steps to use for threshold analysis.  The step size is
        calculated from this by dividing ``1.0/steps``


    Returns
    -------

    measures : pandas.DataFrame

        A pandas dataframe with the following columns:

        * tp: int
        * fp: int
        * tn: int
        * fn: int

    """

    step_size = 1.0 / steps
    data = [
        (index, threshold)
        + sample_measures_for_threshold(pred, gt, mask, threshold)
        for index, threshold in enumerate(numpy.arange(0.0, 1.0, step_size))
    ]

    retval = pandas.DataFrame(
        data,
        columns=(
            "index",
            "threshold",
            "tp",
            "fp",
            "tn",
            "fn",
        ),
    )
    retval.set_index("index", inplace=True)
    return retval


def _sample_analysis(
    img,
    pred,
    gt,
    mask,
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

    mask : torch.Tensor
        region mask (used only if available).  May be set to ``None``.

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

    # if a mask is provided, consider only TP/FP/TN/FN **within** the region of
    # interest defined by the mask
    if mask is not None:
        antimask = torch.le(mask, 0.5)
        tp_tensor[antimask] = 0
        fp_tensor[antimask] = 0
        tn_tensor[antimask] = 0
        fn_tensor[antimask] = 0

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
        # using blend here, to fade original image being overlayed, or
        # its brightness may obfuscate colors from the vessel map
        tp_pil_colored = PIL.Image.blend(img, tp_pil_colored, 0.5)

    return tp_pil_colored


def _summarize(data):
    """Summarizes collected dataframes and adds bayesian figures"""

    _entries = (
        "mean_precision",
        "mode_precision",
        "lower_precision",
        "upper_precision",
        "mean_recall",
        "mode_recall",
        "lower_recall",
        "upper_recall",
        "mean_specificity",
        "mode_specificity",
        "lower_specificity",
        "upper_specificity",
        "mean_accuracy",
        "mode_accuracy",
        "lower_accuracy",
        "upper_accuracy",
        "mean_jaccard",
        "mode_jaccard",
        "lower_jaccard",
        "upper_jaccard",
        "mean_f1_score",
        "mode_f1_score",
        "lower_f1_score",
        "upper_f1_score",
        "frequentist_precision",
        "frequentist_recall",
        "frequentist_specificity",
        "frequentist_accuracy",
        "frequentist_jaccard",
        "frequentist_f1_score",
    )

    def _row_summary(r):

        # run bayesian_measures(), flatten tuple of tuples, name entries
        bayesian = [
            item
            for sublist in bayesian_measures(
                r.tp,
                r.fp,
                r.tn,
                r.fn,
                lambda_=0.5,
                coverage=0.95,
            )
            for item in sublist
        ]

        # evaluate frequentist measures
        frequentist = base_measures(r.tp, r.fp, r.tn, r.fn)
        return pandas.Series(bayesian + list(frequentist), index=_entries)

    # Merges all dataframes together
    sums = pandas.concat(data.values()).groupby("index").sum()
    sums["threshold"] /= len(data)

    # create a new dataframe with these
    measures = sums.apply(lambda r: _row_summary(r), axis=1)

    ## merge sums and measures into a single dataframe
    return pandas.concat([sums, measures.reindex(sums.index)], axis=1).copy()


def run(
    dataset,
    name,
    predictions_folder,
    output_folder=None,
    overlayed_folder=None,
    threshold=None,
    steps=1000,
):
    """
    Runs inference and calculates measures


    Parameters
    ---------

    dataset : py:class:`torch.utils.data.Dataset`
        a dataset to iterate on

    name : str
        the local name of this dataset (e.g. ``train``, or ``test``), to be
        used when saving measures files.

    predictions_folder : str
        folder where predictions for the dataset images has been previously
        stored

    output_folder : :py:class:`str`, Optional
        folder where to store results.  If not provided, then do not store any
        analysis (useful for quickly calculating overlay thresholds)

    overlayed_folder : :py:class:`str`, Optional
        if not ``None``, then it should be the name of a folder where to store
        overlayed versions of the images and ground-truths

    threshold : :py:class:`float`, Optional
        if ``overlayed_folder``, then this should be threshold (floating point)
        to apply to prediction maps to decide on positives and negatives for
        overlaying analysis (graphical output).  This number should come from
        the training set or a separate validation set.  Using a test set value
        may bias your analysis.  This number is also used to print the a priori
        F1-score on the evaluated set.

    steps : :py:class:`float`, Optional
        number of threshold steps to consider when evaluating thresholds.


    Returns
    -------

    threshold : float
        Threshold to achieve the highest possible F1-score for this dataset

    """

    # Collect overall measures
    data = {}

    use_predictions_folder = os.path.join(predictions_folder, name)
    if not os.path.exists(use_predictions_folder):
        use_predictions_folder = predictions_folder

    for sample in tqdm(dataset):
        stem = sample[0]
        image = sample[1]
        gt = sample[2]
        mask = None if len(sample) <= 3 else sample[3]
        pred_fullpath = os.path.join(use_predictions_folder, stem + ".hdf5")
        with h5py.File(pred_fullpath, "r") as f:
            pred = f["array"][:]
        pred = torch.from_numpy(pred)
        if stem in data:
            raise RuntimeError(
                f"{stem} entry already exists in data. Cannot overwrite."
            )
        data[stem] = _sample_measures(pred, gt, mask, steps)

        if output_folder is not None:
            fullpath = os.path.join(output_folder, name, f"{stem}.csv")
            tqdm.write(f"Saving {fullpath}...")
            os.makedirs(os.path.dirname(fullpath), exist_ok=True)
            data[stem].to_csv(fullpath)

        if overlayed_folder is not None:
            overlay_image = _sample_analysis(
                image, pred, gt, mask, threshold=threshold, overlay=True
            )
            fullpath = os.path.join(overlayed_folder, name, f"{stem}.png")
            tqdm.write(f"Saving {fullpath}...")
            os.makedirs(os.path.dirname(fullpath), exist_ok=True)
            overlay_image.save(fullpath)

    # Merges all dataframes together
    measures = _summarize(data)

    maxf1 = measures["mean_f1_score"].max()
    maxf1_index = measures["mean_f1_score"].idxmax()
    maxf1_threshold = measures["threshold"][maxf1_index]

    logger.info(
        f"Maximum F1-score of {maxf1:.5f}, achieved at "
        f"threshold {maxf1_threshold:.3f} (chosen *a posteriori*)"
    )

    if threshold is not None:

        # get the closest possible threshold we have
        index = int(round(steps * threshold))
        f1_a_priori = measures["mean_f1_score"][index]
        actual_threshold = measures["threshold"][index]

        # mark threshold a priori chosen on this dataset
        measures["threshold_a_priori"] = False
        measures["threshold_a_priori", index] = True

        logger.info(
            f"F1-score of {f1_a_priori:.5f}, at threshold "
            f"{actual_threshold:.3f} (chosen *a priori*)"
        )

    if output_folder is not None:
        logger.info(f"Output folder: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        measures_path = os.path.join(output_folder, f"{name}.csv")
        logger.info(
            f"Saving measures over all input images at {measures_path}..."
        )
        measures.to_csv(measures_path)

    return maxf1_threshold


def compare_annotators(
    baseline, other, name, output_folder, overlayed_folder=None
):
    """
    Compares annotations on the **same** dataset


    Parameters
    ---------

    baseline : py:class:`torch.utils.data.Dataset`
        a dataset to iterate on, containing the baseline annotations

    other : py:class:`torch.utils.data.Dataset`
        a second dataset, with the same samples as ``baseline``, but annotated
        by a different annotator than in the first dataset.  The key values
        must much between ``baseline`` and this dataset.

    name : str
        the local name of this dataset (e.g. ``train-second-annotator``, or
        ``test-second-annotator``), to be used when saving measures files.

    output_folder : str
        folder where to store results

    overlayed_folder : :py:class:`str`, Optional
        if not ``None``, then it should be the name of a folder where to store
        overlayed versions of the images and ground-truths

    """

    logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    # Collect overall measures
    data = {}

    for baseline_sample, other_sample in tqdm(
        list(zip(baseline, other)), desc="samples", leave=False, disable=None
    ):
        assert baseline_sample[0] == other_sample[0], (
            f"Mismatch between "
            f"datasets for second-annotator analysis "
            f"({baseline_sample[0]} != {other_sample[0]}).  This "
            f"typically occurs when the second annotator (`other`) "
            f"comes from a different dataset than the `baseline` dataset"
        )

        stem = baseline_sample[0]
        image = baseline_sample[1]
        gt = baseline_sample[2]
        pred = other_sample[2]  # works as a prediction
        mask = None if len(baseline_sample) < 4 else baseline_sample[3]
        if stem in data:
            raise RuntimeError(
                f"{stem} entry already exists in data. " f"Cannot overwrite."
            )
        data[stem] = _sample_measures(pred, gt, mask, 2)

        if output_folder is not None:
            fullpath = os.path.join(
                output_folder, "second-annotator", name, f"{stem}.csv"
            )
            tqdm.write(f"Saving {fullpath}...")
            os.makedirs(os.path.dirname(fullpath), exist_ok=True)
            data[stem].to_csv(fullpath)

        if overlayed_folder is not None:
            overlay_image = _sample_analysis(
                image, pred, gt, mask, threshold=0.5, overlay=True
            )
            fullpath = os.path.join(
                overlayed_folder, "second-annotator", name, f"{stem}.png"
            )
            tqdm.write(f"Saving {fullpath}...")
            os.makedirs(os.path.dirname(fullpath), exist_ok=True)
            overlay_image.save(fullpath)

    measures = _summarize(data)
    measures.drop(0, inplace=True)  #removes threshold == 0.0, keeps 0.5 only

    measures_path = os.path.join(
        output_folder, "second-annotator", f"{name}.csv"
    )
    os.makedirs(os.path.dirname(measures_path), exist_ok=True)
    logger.info(f"Saving summaries over all input images at {measures_path}...")
    measures.to_csv(measures_path)

    maxf1 = measures["mean_f1_score"].max()
    logger.info(f"F1-score of {maxf1:.5f} (second annotator; threshold=0.5)")
