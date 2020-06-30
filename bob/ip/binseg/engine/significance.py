#!/usr/bin/env python
# coding=utf-8

import os
import itertools

import h5py
from tqdm import tqdm
import pandas
import torch.nn

from .evaluator import _sample_measures_for_threshold


def _patch_measures(pred, gt, threshold, size, stride):
    """
    Calculates measures on patches of a single sample


    Parameters
    ----------

    pred : torch.Tensor
        pixel-wise predictions

    gt : torch.Tensor
        ground-truth (annotations)

    threshold : float
        threshold to use for evaluating individual patch performances

    size : tuple
        size (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth

    stride : tuple
        strides (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth


    Returns
    -------

    measures : pandas.DataFrame

        A pandas dataframe with the following columns:

        * patch: int
        * threshold: float
        * precision: float
        * recall: float
        * specificity: float
        * accuracy: float
        * jaccard: float
        * f1_score: float

    """

    # we calculate the required padding so that the last windows on the left
    # and bottom size of predictions/ground-truth data are zero padded, and
    # torch unfolding works exactly.
    padding = (0, 0)
    rem = (pred.shape[1] - size[1]) % stride[1]
    if rem != 0:
        padding = (0, (stride[1] - rem))
    rem = (pred.shape[0] - size[0]) % stride[0]
    if rem != 0:
        padding += (0, (stride[0] - rem))

    pred_padded = torch.nn.functional.pad(pred, padding)
    gt_padded = torch.nn.functional.pad(gt.squeeze(0), padding)

    # this will create as many views as required
    pred_patches = pred_padded.unfold(0, size[0], stride[0]).unfold(
        1, size[1], stride[1]
    )
    gt_patches = gt_padded.unfold(0, size[0], stride[0]).unfold(
        1, size[1], stride[1]
    )
    assert pred_patches.shape == gt_patches.shape
    ylen, xlen, _, _ = pred_patches.shape

    data = [
        [j, i]
        + _sample_measures_for_threshold(
            pred_patches[j, i, :, :], gt_patches[j, i, :, :], threshold
        )
        for j, i in itertools.product(range(ylen), range(xlen))
    ]

    return pandas.DataFrame(
        data,
        columns=(
            "y",
            "x",
            "precision",
            "recall",
            "specificity",
            "accuracy",
            "jaccard",
            "f1_score",
        ),
    )


def patch_performances(
    dataset, name, predictions_folder, threshold, size, stride
):
    """
    Evaluates the performances for multiple image patches, for a whole dataset


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

    threshold : :py:class:`float`
        this should be a threshold (floating point) to apply to prediction maps
        to decide on positives and negatives.

    size : tuple
        size (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth

    stride : tuple
        strides (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth


    Returns
    -------

    df : pandas.DataFrame
        A dataframe with all the patch performances aggregated, for all input
        images.

    """

    # Collect overall measures
    data = []

    use_predictions_folder = os.path.join(predictions_folder, name)
    if not os.path.exists(use_predictions_folder):
        use_predictions_folder = predictions_folder

    for sample in tqdm(dataset[name]):
        stem = sample[0]
        image = sample[1]
        gt = sample[2]
        pred_fullpath = os.path.join(use_predictions_folder, stem + ".hdf5")
        with h5py.File(pred_fullpath, "r") as f:
            pred = f["array"][:]
        pred = torch.from_numpy(pred)
        data.append(_patch_measures(pred, gt, threshold, size, stride))
        data[-1]["stem"] = stem

    return pandas.concat(data, ignore_index=True)
