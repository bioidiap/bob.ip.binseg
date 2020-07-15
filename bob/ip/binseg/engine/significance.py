#!/usr/bin/env python
# coding=utf-8

import os
import itertools
import multiprocessing

import h5py
from tqdm import tqdm
import numpy
import pandas
import torch.nn

from .evaluator import _sample_measures_for_threshold


def _performance_summary(size, patch_perf, patch_size, patch_stride, figure):
    """Generates an array that represents the performance per pixel of the
    original image

    The returned array corresponds to a stacked version of performances for
    each pixel in the original image taking into consideration the patch
    performances, their size and stride.


    Parameters
    ==========

    size : tuple
        A two tuple with the original height and width of the image being
        analyzed

    patch_perf : typing.Sequence
        An ordered sequence of patch performances (in raster direction - every
        row, from left to right and then rows from top to bottom).

    patch_size : tuple
        A two tuple that indicates the size of each patch (height, width)

    patch_stride: tuple
        A two tuple that indicates the stride of each patch (height, width)

    figure: str
        Name of the performance figure to use for the summary


    Returns
    =======

    n : numpy.ndarray
        A 2D numpy array containing the number of performance scores for every
        pixel in the original image

    avg : numpy.ndarray
        A 2D numpy array containing the average performances for every pixel on
        the input image considering the patch sizes and strides applied when
        windowing the image

    std : numpy.ndarray
        A 2D numpy array containing the (unbiased) standard deviations for the
        provided performance figure, for every pixel on the input image
        considering the patch sizes and strides applied when windowing the
        image

    """

    # first, estimate how many overlaps we have per pixel in the original image

    # we calculate the required padding so that the last windows on the left
    # and bottom size of predictions/ground-truth data are zero padded, and
    # torch unfolding works exactly.  The last windows on the left and bottom
    # parts of the image may be extended with zeros.
    final_size = list(size)
    rem = (size[0] - patch_size[0]) % patch_stride[0]
    if rem != 0:
        final_size[0] += patch_stride[0] - rem
    rem = (size[1] - patch_size[1]) % patch_stride[1]
    if rem != 0:
        final_size[1] += patch_stride[1] - rem
    n = numpy.zeros(final_size, dtype=int)
    ylen = ((final_size[0] - patch_size[0]) // patch_stride[0]) + 1
    xlen = ((final_size[1] - patch_size[1]) // patch_stride[1]) + 1
    for j in range(ylen):
        for i in range(xlen):
            yup = slice(
                patch_stride[0] * j, (patch_stride[0] * j) + patch_size[0], 1
            )
            xup = slice(
                patch_stride[1] * i, (patch_stride[1] * i) + patch_size[1], 1
            )
            n[yup, xup] += 1

    # calculates the stacked performance
    layers = n.max()
    perf = numpy.zeros(
        [layers] + final_size, dtype=patch_perf[figure].iloc[0].dtype
    )
    n = -1 * numpy.ones(final_size, dtype=int)
    for j in range(ylen):
        for i in range(xlen):
            yup = slice(
                patch_stride[0] * j, (patch_stride[0] * j) + patch_size[0], 1
            )
            xup = slice(
                patch_stride[1] * i, (patch_stride[1] * i) + patch_size[1], 1
            )
            nup = n[yup, xup]
            nup += 1
            yr, xr = numpy.meshgrid(
                range(yup.start, yup.stop, yup.step),
                range(xup.start, xup.stop, xup.step),
                indexing="ij",
            )
            perf[nup.flat, yr.flat, xr.flat] = patch_perf.loc[
                (patch_perf["y"] == j) & (patch_perf["x"] == i)
            ][figure]

    # for each element in the ``perf``matrix, calculated avg and std.
    n += 1  # adjust for starting at -1 before
    avg = perf.sum(axis=0) / n
    # calculate variances
    std = numpy.zeros_like(avg)
    for k in range(2, n.max() + 1):
        std[n == k] = perf[: (k + 1), :, :].std(axis=0, ddof=1)[n == k]

    return n, avg, std


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
    # torch unfolding works exactly.  The last windows on the left and bottom
    # parts of the image may be extended with zeros.
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


def _visual_dataset_performance(stem, img, n, avg, std, outdir):
    """Runs a visual performance assessment for each entry in a given dataset


    Parameters
    ----------

    stem : str
        The input file stem, for which a figure will be saved in ``outdir``,
        in PDF format

    img : pytorch.Tensor
        A 3D tensor containing the original image that was analyzed

    n : numpy.ndarray
        A 2D integer array with the same size as `img` that indicates how many
        overlapping windows are available for each pixel in the image

    avg : numpy.ndarray
        A 2D floating-point array with the average performance per pixel
        calculated over all overlapping windows for that particular pixel

    std : numpy.ndarray
        A 2D floating-point array with the unbiased standard-deviation
        (``ddof=1``) performance per pixel calculated over all overlapping
        windows for that particular pixel

    outdir : str
        The base directory where to save output PDF images generated by this
        procedure.  The value of ``stem`` will be suffixed to this output
        directory using a standard path join.  The output filename will have a
        ``.pdf`` extension.

    """

    import matplotlib.pyplot as plt

    figsize = img.shape[1:]

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(img.numpy().transpose(1, 2, 0))
    axs[0, 0].set_title("Original image")

    pos01 = axs[0, 1].imshow(
        n[: figsize[0], : figsize[1]], cmap="Greys_r", interpolation="none"
    )
    fig.colorbar(pos01, ax=axs[0, 1], orientation="vertical")
    axs[0, 1].set_title("# Overlapping Windows")

    pos10 = axs[1, 0].imshow(
        avg[: figsize[0], : figsize[1]], cmap="Greys_r", interpolation="none"
    )
    fig.colorbar(pos10, ax=axs[1, 0], orientation="vertical")
    axs[1, 0].set_title("Average Performance")

    pos11 = axs[1, 1].imshow(
        std[: figsize[0], : figsize[1]], cmap="Greys_r", interpolation="none"
    )
    fig.colorbar(pos11, ax=axs[1, 1], orientation="vertical")
    axs[1, 1].set_title("Performance Std. Dev.")

    plt.tight_layout()

    fname = os.path.join(outdir, stem + ".pdf")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname)


def _patch_performances_for_sample(
    basedir, threshold, size, stride, dataset, k, figure, outdir,
):
    """
    Evaluates patch performances per sample


    Parameters
    ----------

    basedir : str
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

    dataset : :py:class:`dict` of py:class:`torch.utils.data.Dataset`
        datasets to iterate on

    k : int
        the sample number (order inside the dataset, starting from zero), to
        calculate patch performances for

    figure : str
        the performance figure to use for calculating patch micro performances
        (e.g. `f1_score` or `jaccard`).  Must be available on the produced
        performance dataframe.

    outdir : :py:class:`str`
        path were to save a visual representation of patch performances.  If
        set to ``None``, then do not save those to disk.


    Returns
    -------

    stem : str
        The input file stem, that was just analyzed

    data : dict
        A dictionary containing the following fields:

        * ``df``: a :py:class:`pandas.DataFrame` with the patch performance
          figures in raster scan order.
        * ``n``: a 2D integer :py:class:`numpy.ndarray` with the same size as
          the original image pertaining to the analyzed sample, that indicates
          how many overlapping windows are available for each pixel in the
          image
        * ``avg``: a 2D floating-point :py:class:`numpy.ndarray` with the
          average performance per pixel calculated over all overlapping windows
          for that particular pixel
        * ``std``: a 2D floating-point :py:class:`numpy.ndarray` with the
          unbiased standard-deviation (``ddof=1``) performance per pixel
          calculated over all overlapping windows for that particular pixel

    """

    sample = dataset[k]
    with h5py.File(os.path.join(basedir, sample[0] + ".hdf5"), "r") as f:
        pred = torch.from_numpy(f["array"][:])
    df = _patch_measures(pred, sample[2], threshold, size, stride)
    n, avg, std = _performance_summary(
        sample[1].shape[1:], df, size, stride, figure
    )
    if outdir is not None:
        _visual_dataset_performance(sample[0], sample[1], n, avg, std, outdir)
    return sample[0], dict(df=df, n=n, avg=avg, std=std)


def patch_performances(
    dataset,
    name,
    predictions_folder,
    threshold,
    size,
    stride,
    figure,
    nproc=1,
    outdir=None,
):
    """
    Evaluates the performances for multiple image patches, for a whole dataset


    Parameters
    ---------

    dataset : :py:class:`dict` of py:class:`torch.utils.data.Dataset`
        datasets to iterate on

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

    figure : str
        the performance figure to use for calculating patch micro performances

    nproc : :py:class:`int`, Optional
        the number of processing cores to use for performance evaluation.
        Setting this number to a value ``<= 0`` will cause the function to use
        all available computing cores.  Otherwise, limit to the number of cores
        specified.  If set to the value of ``1``, then do not use
        multiprocessing.

    outdir : :py:class:`str`, Optional
        path were to save a visual representation of patch performances.  If
        set to ``None``, then do not save those to disk.


    Returns
    -------

    d : dict
        A dictionary in which keys are filename stems and values are
        dictionaries with the following contents:

        ``df``: :py:class:`pandas.DataFrame`
            A dataframe with all the patch performances aggregated, for all
            input images.

        ``n`` : :py:class:`numpy.ndarray`
            A 2D numpy array containing the number of performance scores for
            every pixel in the original image

        ``avg`` : :py:class:`numpy.ndarray`
            A 2D numpy array containing the average performances for every
            pixel on the input image considering the patch sizes and strides
            applied when windowing the image

        ``std`` : :py:class:`numpy.ndarray`
            A 2D numpy array containing the (unbiased) standard deviations for
            the provided performance figure, for every pixel on the input image
            considering the patch sizes and strides applied when windowing the
            image

    """

    use_predictions_folder = os.path.join(predictions_folder, name)
    if not os.path.exists(use_predictions_folder):
        use_predictions_folder = predictions_folder

    with tqdm(range(len(dataset[name])), desc="patch-perf") as pbar:
        # we avoid the multiprocessing module if nproc==1
        # so it is easier to run ipdb
        if nproc != 1:
            if nproc <= 0:
                nproc = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(nproc)
            results = [
                pool.apply_async(
                    _patch_performances_for_sample,
                    args=(
                        use_predictions_folder,
                        threshold,
                        size,
                        stride,
                        dataset[name],
                        k,
                        figure,
                        outdir,
                    ),
                    callback=lambda x: pbar.update(),
                )
                for k in range(len(dataset[name]))
            ]
            pool.close()
            pool.join()
            data = [k.get() for k in results]
        else:
            data = []
            for k in pbar:
                df = _patch_performances_for_sample(
                    use_predictions_folder,
                    threshold,
                    size,
                    stride,
                    dataset[name],
                    k,
                    figure,
                    outdir,
                )
                data.append(df)

    return dict(data)


def _visual_performances_for_sample(size, stride, dataset, k, df, figure, outdir):
    """
    Displays patch performances per sample

    This is a simplified version of :py:func:`_patch_performances_for_sample`
    in which the patch performances are not recalculated and used as input.  It
    can be used in case you have the patch performances stored in disk or if
    you're evaluating differences between patches of 2 different systems.


    Parameters
    ----------

    size : tuple
        size (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth

    stride : tuple
        strides (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth

    dataset : :py:class:`dict` of py:class:`torch.utils.data.Dataset`
        datasets to iterate on

    k : int
        the sample number (order inside the dataset, starting from zero), to
        calculate patch performances for

    df : pandas.DataFrame
        the previously calculated dataframe to use for this patch performance
        assessment.

    figure : str
        the performance figure to use for calculating patch micro performances
        (e.g. `f1_score` or `jaccard`).  Must be available on the produced
        performance dataframe.

    outdir : :py:class:`str`
        path were to save a visual representation of patch performances.  If
        set to ``None``, then do not save those to disk.


    Returns
    -------

    stem : str
        The input file stem, that was just analyzed

    data : dict
        A dictionary containing the following fields:

        * ``df``: a :py:class:`pandas.DataFrame` with the patch performance
          figures in raster scan order.  Notice this is just a copy of the
          input data frame with the same name.
        * ``n``: a 2D integer :py:class:`numpy.ndarray` with the same size as
          the original image pertaining to the analyzed sample, that indicates
          how many overlapping windows are available for each pixel in the
          image
        * ``avg``: a 2D floating-point :py:class:`numpy.ndarray` with the
          average performance per pixel calculated over all overlapping windows
          for that particular pixel
        * ``std``: a 2D floating-point :py:class:`numpy.ndarray` with the
          unbiased standard-deviation (``ddof=1``) performance per pixel
          calculated over all overlapping windows for that particular pixel

    """

    sample = dataset[k]
    n, avg, std = _performance_summary(
        sample[1].shape[1:], df, size, stride, figure
    )
    if outdir is not None:
        _visual_dataset_performance(sample[0], sample[1], n, avg, std, outdir)
    return sample[0], dict(df=df, n=n, avg=avg, std=std)


def visual_performances(
    dataset, name, dfs, size, stride, figure, nproc=1, outdir=None,
):
    """
    Displays the performances for multiple image patches, for a whole dataset

    This is a simplified version of :py:func:`patch_performances` in which the
    patch performances are not recalculated and used as input.  It can be used
    in case you have the patch performances stored in disk or if you're
    evaluating differences between patches of 2 different systems.


    Parameters
    ---------

    dataset : :py:class:`dict` of py:class:`torch.utils.data.Dataset`
        datasets to iterate on

    name : str
        the local name of this dataset (e.g. ``train``, or ``test``), to be
        used when saving measures files.

    dfs : dict
        a dictionary mapping dataset stems to dataframes containing the patch
        performances to be evaluated

    size : tuple
        size (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth

    stride : tuple
        strides (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth

    figure : str
        the performance figure to use for calculating patch micro performances

    nproc : :py:class:`int`, Optional
        the number of processing cores to use for performance evaluation.
        Setting this number to a value ``<= 0`` will cause the function to use
        all available computing cores.  Otherwise, limit to the number of cores
        specified.  If set to the value of ``1``, then do not use
        multiprocessing.

    outdir : :py:class:`str`, Optional
        path were to save a visual representation of patch performances.  If
        set to ``None``, then do not save those to disk.


    Returns
    -------

    d : dict
        A dictionary in which keys are filename stems and values are
        dictionaries with the following contents:

        ``df``: :py:class:`pandas.DataFrame`
            A dataframe with all the patch performances aggregated, for all
            input images.

        ``n`` : :py:class:`numpy.ndarray`
            A 2D numpy array containing the number of performance scores for
            every pixel in the original image

        ``avg`` : :py:class:`numpy.ndarray`
            A 2D numpy array containing the average performances for every
            pixel on the input image considering the patch sizes and strides
            applied when windowing the image

        ``std`` : :py:class:`numpy.ndarray`
            A 2D numpy array containing the (unbiased) standard deviations for
            the provided performance figure, for every pixel on the input image
            considering the patch sizes and strides applied when windowing the
            image

    """

    stems = list(dataset[name].keys())

    with tqdm(range(len(dataset[name])), desc="visual-perf") as pbar:
        # we avoid the multiprocessing module if nproc==1
        # so it is easier to run ipdb
        if nproc != 1:
            if nproc <= 0:
                nproc = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(nproc)
            results = [
                pool.apply_async(
                    _visual_performances_for_sample,
                    args=(
                        size,
                        stride,
                        dataset[name],
                        k,
                        dfs[stems[k]],
                        figure,
                        outdir,
                    ),
                    callback=lambda x: pbar.update(),
                )
                for k in range(len(dataset[name]))
            ]
            pool.close()
            pool.join()
            data = [k.get() for k in results]
        else:
            data = []
            for k in pbar:
                df = _visual_performances_for_sample(
                    size, stride, dataset[name], k, dfs[stems[k]], figure, outdir,
                )
                data.append(df)

    return dict(data)
