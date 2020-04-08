#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines functionality for the evaluation of predictions"""

import os

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


def _sample_metrics(stem, pred, gt):
    """
    Calculates metrics on one single sample and saves it to disk


    Parameters
    ----------

    stem : str
        original filename without extension and relative to its root-path

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
    gts = gt.byte()

    data = []

    for threshold in numpy.arange(0.0, 1.0, step_size):

        # threshold
        binary_pred = torch.gt(pred, threshold).byte()

        # equals and not-equals
        equals = torch.eq(binary_pred, gts).type(torch.uint8)  # tensor
        notequals = torch.ne(binary_pred, gts).type(torch.uint8)  # tensor

        # true positives
        tp_tensor = gt * binary_pred  # tensor
        tp_count = torch.sum(tp_tensor).item()  # scalar

        # false positives
        fp_tensor = torch.eq((binary_pred + tp_tensor), 1)
        fp_count = torch.sum(fp_tensor).item()

        # true negatives
        tn_tensor = equals - tp_tensor
        tn_count = torch.sum(tn_tensor).item()

        # false negatives
        fn_tensor = notequals - fp_tensor.type(torch.uint8)
        fn_count = torch.sum(fn_tensor).item()

        # calc metrics
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


def run(data_loader, predictions_folder, output_folder):
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
        data[stem] = _sample_metrics(stem, pred, gt)

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
