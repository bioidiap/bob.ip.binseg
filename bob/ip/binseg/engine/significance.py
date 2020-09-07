#!/usr/bin/env python
# coding=utf-8

import os
import textwrap
import multiprocessing

import logging

logger = logging.getLogger(__name__)

import h5py
from tqdm import tqdm
import numpy
import torch.nn
import scipy.stats
import tabulate

from .evaluator import sample_measures_for_threshold


PERFORMANCE_FIGURES = [
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "jaccard",
    "f1_score",
]
"""List of performance figures supported by this module, in order"""


def _performance_summary(size, winperf, winsize, winstride, figure):
    """Generates an array that represents the performance per pixel of the
    original image

    The returned array corresponds to a stacked version of performances for
    each pixel in the original image taking into consideration the sliding
    window performances, their size and stride.


    Parameters
    ==========

    size : tuple
        A two tuple with the original height and width of the image being
        analyzed

    winperf : numpy.ndarray
        A 3D array with shape ``(N, H, W)``, where ``N`` represents the number
        of performance measures supported by this module, ``(H,W)`` is the
        total number of vertical and horizontal sliding windows.

    winsize : tuple
        A two tuple that indicates the size of the sliding window (height,
        width)

    winstride : tuple
        A two tuple that indicates the stride of the sliding window (height,
        width)

    figure : str
        Name of the performance figure to use for the summary


    Returns
    =======

    n : numpy.ndarray
        A 2D numpy array containing the number of performance scores for every
        pixel in the original image

    avg : numpy.ndarray
        A 2D numpy array containing the average performances for every pixel on
        the input image considering the sliding window sizes and strides
        applied to the image

    std : numpy.ndarray
        A 2D numpy array containing the (unbiased) standard deviations for the
        provided performance figure, for every pixel on the input image
        considering the sliding window sizes and strides applied to the image

    """

    # first, estimate how many overlaps we have per pixel in the original image

    # we calculate the required padding so that the last windows on the left
    # and bottom size of predictions/ground-truth data are zero padded, and
    # torch unfolding works exactly.  The last windows on the left and bottom
    # parts of the image may be extended with zeros.
    final_size = list(size)
    rem = (size[0] - winsize[0]) % winstride[0]
    if rem != 0:
        final_size[0] += winstride[0] - rem
    rem = (size[1] - winsize[1]) % winstride[1]
    if rem != 0:
        final_size[1] += winstride[1] - rem
    n = numpy.zeros(final_size, dtype=int)

    # calculates the stacked performance
    layers = int(
        numpy.ceil(winsize[0] / winstride[0])
        * numpy.ceil(winsize[1] / winstride[1])
    )
    figindex = PERFORMANCE_FIGURES.index(figure)
    perf = numpy.zeros([layers] + final_size, dtype=winperf.dtype)
    n = -1 * numpy.ones(final_size, dtype=int)
    data = winperf[PERFORMANCE_FIGURES.index(figure)]
    for j in range(data.shape[0]):
        yup = slice(winstride[0] * j, (winstride[0] * j) + winsize[0], 1)
        for i in range(data.shape[1]):
            xup = slice(winstride[1] * i, (winstride[1] * i) + winsize[1], 1)
            nup = n[yup, xup]
            nup += 1
            yr, xr = numpy.meshgrid(
                range(yup.start, yup.stop, yup.step),
                range(xup.start, xup.stop, xup.step),
                indexing="ij",
            )
            perf[nup.flat, yr.flat, xr.flat] = data[j, i]

    # for each element in the ``perf``matrix, calculates avg and std.
    n += 1  # adjust for starting at -1 before
    avg = perf.sum(axis=0) / n
    # calculate variances
    std = numpy.zeros_like(avg)
    for k in range(2, n.max() + 1):
        std[n == k] = perf[:k].std(axis=0, ddof=1)[n == k]

    # returns only valid bounds wrt to the original image
    return (
        n[: size[0], : size[1]],
        avg[: size[0], : size[1]],
        std[: size[0], : size[1]],
    )


def _winperf_measures(pred, gt, mask, threshold, size, stride):
    """
    Calculates measures on sliding windows of a single sample


    Parameters
    ----------

    pred : torch.Tensor
        pixel-wise predictions

    gt : torch.Tensor
        ground-truth (annotations)

    mask : torch.Tensor
        mask for the region of interest

    threshold : float
        threshold to use for evaluating individual sliding window performances

    size : tuple
        size (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth

    stride : tuple
        strides (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth


    Returns
    -------

    measures : numpy.ndarray

        A 3D float array with all supported performance entries for each
        sliding window.

        The first dimension of the array is therefore 6.  The other two
        dimensions correspond to resulting size of the sliding window operation
        applied to the input data and taking into consideration the sliding
        window size and the stride.

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

    pred_padded = torch.nn.functional.pad(
        pred, padding, mode="constant", value=0.0
    )
    gt_padded = torch.nn.functional.pad(
        gt.squeeze(0), padding, mode="constant", value=0.0
    )
    mask_padded = torch.nn.functional.pad(
        mask.squeeze(0), padding, mode="constant", value=1.0
    )

    # this will create as many views as required
    pred_windows = pred_padded.unfold(0, size[0], stride[0]).unfold(
        1, size[1], stride[1]
    )
    gt_windows = gt_padded.unfold(0, size[0], stride[0]).unfold(
        1, size[1], stride[1]
    )
    mask_windows = mask_padded.unfold(0, size[0], stride[0]).unfold(
        1, size[1], stride[1]
    )
    assert pred_windows.shape == gt_windows.shape
    assert gt_windows.shape == mask_windows.shape
    ylen, xlen, _, _ = pred_windows.shape

    retval = numpy.array(
        [
            sample_measures_for_threshold(
                pred_windows[j, i, :, :],
                gt_windows[j, i, :, :],
                mask_windows[j, i, :, :],
                threshold,
            )
            for j in range(ylen)
            for i in range(xlen)
        ]
    )
    return retval.transpose(1, 0).reshape(6, ylen, xlen)


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
    plt.close(fig)


def _winperf_for_sample(
    basedir, threshold, size, stride, dataset, k, figure, outdir,
):
    """
    Evaluates sliding window performances per sample


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
        calculate sliding window performances for

    figure : str
        the performance figure to use for calculating sliding window micro
        performances (e.g. `accuracy`, `f1_score` or `jaccard`).  Must be
        a supported performance figure as defined in
        :py:attr:`PERFORMANCE_FIGURES`

    outdir : str
        path were to save a visual representation of sliding window
        performances.  If set to ``None``, then do not save those to disk.


    Returns
    -------

    stem : str
        The input file stem, that was just analyzed

    data : dict
        A dictionary containing the following fields:

        * ``winperf``: a 3D :py:class:`numpy.ndarray` with the sliding window
          performance figures
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
    mask = None if len(sample) < 4 else sample[3]
    winperf = _winperf_measures(pred, sample[2], mask, threshold, size, stride)
    n, avg, std = _performance_summary(
        sample[1].shape[1:], winperf, size, stride, figure
    )
    if outdir is not None:
        _visual_dataset_performance(sample[0], sample[1], n, avg, std, outdir)
    return sample[0], dict(winperf=winperf, n=n, avg=avg, std=std)


def sliding_window_performances(
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
    Evaluates sliding window performances for a whole dataset


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
        the performance figure to use for calculating sliding window micro
        performances

    nproc : :py:class:`int`, Optional
        the number of processing cores to use for performance evaluation.
        Setting this number to a value ``<= 0`` will cause the function to use
        all available computing cores.  Otherwise, limit to the number of cores
        specified.  If set to the value of ``1``, then do not use
        multiprocessing.

    outdir : :py:class:`str`, Optional
        path were to save a visual representation of sliding window
        performances.  If set to ``None``, then do not save those to disk.


    Returns
    -------

    d : dict
        A dictionary in which keys are filename stems and values are
        dictionaries with the following contents:

        ``winperf``: numpy.ndarray

        ``n`` : numpy.ndarray
            A 2D numpy array containing the number of performance scores for
            every pixel in the original image

        ``avg`` : :py:class:`numpy.ndarray`
            A 2D numpy array containing the average performances for every
            pixel on the input image considering the sliding window sizes and
            strides applied to the image

        ``std`` : :py:class:`numpy.ndarray`
            A 2D numpy array containing the (unbiased) standard deviations for
            the provided performance figure, for every pixel on the input image
            considering the sliding window sizes and strides applied to the
            image

    """

    use_predictions_folder = os.path.join(predictions_folder, name)
    if not os.path.exists(use_predictions_folder):
        use_predictions_folder = predictions_folder

    with tqdm(range(len(dataset[name])), desc="sld-win-perf") as pbar:
        # we avoid the multiprocessing module if nproc==1
        # so it is easier to run ipdb
        if nproc != 1:
            if nproc <= 0:
                nproc = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(nproc)
            results = [
                pool.apply_async(
                    _winperf_for_sample,
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
                winperf = _winperf_for_sample(
                    use_predictions_folder,
                    threshold,
                    size,
                    stride,
                    dataset[name],
                    k,
                    figure,
                    outdir,
                )
                data.append(winperf)

    return dict(data)


def _visual_performances_for_sample(
    size, stride, dataset, k, winperf, figure, outdir
):
    """
    Displays sliding windows performances per sample

    This is a simplified version of :py:func:`_winper_for_sample`
    in which the sliding window performances are not recalculated and used as
    input.  It can be used in case you have the sliding window performances
    stored in disk or if you're evaluating differences between sliding windows
    of 2 different systems.


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
        calculate sliding window performances for

    winperf : numpy.ndarray
        the previously calculated sliding window performances to use for this
        assessment.

    figure : str
        the performance figure to use for calculating sliding window micro
        performances (e.g. `accuracy`, `f1_score` or `jaccard`).  Must be
        a supported performance figure as defined in
        :py:attr:`PERFORMANCE_FIGURES`

    outdir : :py:class:`str`
        path were to save a visual representation of sliding window
        performances.  If set to ``None``, then do not save those to disk.


    Returns
    -------

    stem : str
        The input file stem, that was just analyzed

    data : dict
        A dictionary containing the following fields:

        * ``winperf``: a 3D float :py:class:`numpy.ndarray` with the sliding
          window performance figures.  Notice this is just a copy of the input
          sliding window performance figures with the same name.
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
        sample[1].shape[1:], winperf, size, stride, figure
    )
    if outdir is not None:
        _visual_dataset_performance(sample[0], sample[1], n, avg, std, outdir)
    return sample[0], dict(winperf=winperf, n=n, avg=avg, std=std)


def visual_performances(
    dataset, name, winperfs, size, stride, figure, nproc=1, outdir=None,
):
    """
    Displays the performances for for a whole dataset

    This is a simplified version of :py:func:`sliding_window_performances` in
    which the sliding window performances are not recalculated and used as
    input.  It can be used in case you have the sliding window performances
    stored in disk or if you're evaluating differences between sliding windows
    of 2 different systems.


    Parameters
    ---------

    dataset : :py:class:`dict` of py:class:`torch.utils.data.Dataset`
        datasets to iterate on

    name : str
        the local name of this dataset (e.g. ``train``, or ``test``), to be
        used when saving measures files.

    winperfs : dict
        a dictionary mapping dataset stems to arrays containing the sliding
        window performances to be evaluated

    size : tuple
        size (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth

    stride : tuple
        strides (vertical, horizontal) for windows for which we will calculate
        partial performances based on the threshold and existing ground-truth

    figure : str
        the performance figure to use for calculating sliding window micro
        performances

    nproc : :py:class:`int`, Optional
        the number of processing cores to use for performance evaluation.
        Setting this number to a value ``<= 0`` will cause the function to use
        all available computing cores.  Otherwise, limit to the number of cores
        specified.  If set to the value of ``1``, then do not use
        multiprocessing.

    outdir : :py:class:`str`, Optional
        path were to save a visual representation of sliding window
        performances.  If set to ``None``, then do not save those to disk.


    Returns
    -------

    d : dict
        A dictionary in which keys are filename stems and values are
        dictionaries with the following contents:

        ``winperf``: numpy.ndarray
            A 3D float array with all the sliding window performances for the
            input image

        ``n`` : numpy.ndarray
            A 2D numpy array containing the number of performance scores for
            every pixel in the original image

        ``avg`` : numpy.ndarray
            A 2D numpy array containing the average performances for every
            pixel on the input image considering the sliding window sizes and
            strides applied to the image

        ``std`` : :py:class:`numpy.ndarray`
            A 2D numpy array containing the (unbiased) standard deviations for
            the provided performance figure, for every pixel on the input image
            considering the sliding window sizes and strides applied to the
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
                        winperfs[stems[k]],
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
                winperf = _visual_performances_for_sample(
                    size,
                    stride,
                    dataset[name],
                    k,
                    winperfs[stems[k]],
                    figure,
                    outdir,
                )
                data.append(winperf)

    return dict(data)


def index_of_outliers(c):
    """Finds indexes of outliers (+/- 1.5*IQR) on an array of random values

    The IQR measures the midspread or where 50% of a normal distribution would
    sit, if the input data is, indeed, normal.  1.5 IQR corresponds to a
    symmetrical range that would encompass most of the data, characterizing
    outliers (outside of that range).  Check out `this Wikipedia page
    <https://en.wikipedia.org/wiki/Interquartile_range>` for more details.


    Parameters
    ----------

    c : numpy.ndarray
        A 1D float array


    Returns
    -------

    indexes : numpy.ndarray
        Indexes of the input column that are considered outliers in the
        distribution (outside the 1.5 Interquartile Range).

    """

    l, h = numpy.quantile(c, (0.25, 0.75))
    iqr = h - l
    limits = (l - 1.5 * iqr, h + 1.5 * iqr)
    return (c < limits[0]) | (c > limits[1])


def write_analysis_text(names, da, db, f):
    """Writes a text file containing the most important statistics

    Compares sliding window performances in ``da`` and ``db`` taking into
    consideration their statistical properties.  A significance test is applied
    to check whether observed differences in the statistics of both
    distributions is significant.


    Parameters
    ==========

    names : tuple
        A tuple containing two strings which are the names of the systems being
        analyzed

    da : numpy.ndarray
        A 1D numpy array containing all the performance figures per sliding
        window analyzed and organized in a particular order (raster), for the
        first system (first entry of ``names``)

    db : numpy.ndarray
        A 1D numpy array containing all the performance figures per sliding
        window analyzed and organized in a particular order (raster), for the
        second system (second entry of ``names``)

    f : file
        An open file that will be used dump the analysis to

    """

    diff = da - db
    f.write("Basic statistics from distributions:\n")

    headers = [
        "system",
        "samples",
        "median",
        "average",
        "std.dev.",
        "normaltest (p)",
    ]
    table = [
        [
            names[0],
            len(da),
            numpy.median(da),
            numpy.mean(da),
            numpy.std(da, ddof=1),
            scipy.stats.normaltest(da)[1],
        ],
        [
            names[1],
            len(db),
            numpy.median(db),
            numpy.mean(db),
            numpy.std(db, ddof=1),
            scipy.stats.normaltest(db)[1],
        ],
        [
            "differences",
            len(diff),
            numpy.median(diff),
            numpy.mean(diff),
            numpy.std(diff, ddof=1),
            scipy.stats.normaltest(diff)[1],
        ],
    ]
    tdata = tabulate.tabulate(table, headers, tablefmt="rst", floatfmt=".3f")
    f.write(textwrap.indent(tdata, "  "))
    f.write("\n")

    # Note: dependent variable = sliding window performance figure in our case
    # Assumptions of a Paired T-test:
    # * The dependent variable must be continuous (interval/ratio). [OK]
    # * The observations are independent of one another. [OK]
    # * The dependent variable should be approximately normally distributed. [!!!]
    # * The dependent variable should not contain any outliers. [OK]

    if (diff == 0.0).all():
        logger.error(
            "Differences are exactly zero between both "
            "sliding window distributions, for **all** samples. "
            "Statistical significance tests are not meaningful in "
            "this context and will be skipped.  This typically "
            "indicates an issue with the setup of prediction folders "
            "(duplicated?)"
        )
        return

    f.write("\nPaired significance tests:\n")
    w, p = scipy.stats.ttest_rel(da, db)
    f.write(f"  * Paired T (H0: same distro): S = {w:g}, p = {p:.5f}\n")

    f.write("  * Wilcoxon:\n")

    w, p = scipy.stats.wilcoxon(diff)
    f.write(f"    * H0 = same distro: W = {w:g}, p = {p:.5f}\n")

    w, p = scipy.stats.wilcoxon(diff, alternative="greater")
    f.write(
        f"    * H0 = med({names[0]}) < med({names[1]}): "
        f"W = {w:g}, p = {p:.5f}\n"
    )

    w, p = scipy.stats.wilcoxon(diff, alternative="less")
    f.write(
        f"    * H0 = med({names[0]}) > med({names[1]}): "
        f"W = {w:g}, p = {p:.5f}\n"
    )


def write_analysis_figures(names, da, db, fname):
    """Writes a PDF containing most important plots for analysis


    Parameters
    ==========

    names : tuple
        A tuple containing two strings which are the names of the systems being
        analyzed

    da : numpy.ndarray
        A 1D numpy array containing all the performance figures per sliding
        window analyzed and organized in a particular order (raster), for the
        first system (first entry of ``names``)

    db : numpy.ndarray
        A 1D numpy array containing all the performance figures per sliding
        window analyzed and organized in a particular order (raster), for the
        second system (second entry of ``names``)

    fname : str
        The filename to use for storing the summarized performance figures

    """

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    diff = da - db
    bins = 50

    with PdfPages(fname) as pdf:

        fig = plt.figure()
        plt.grid()
        plt.hist(da, bins=bins)
        plt.title(
            f"{names[0]} - (N={len(da)}; M={numpy.median(da):.3f}; "
            f"$\mu$={numpy.mean(da):.3f}; $\sigma$={numpy.std(da, ddof=1):.3f})"
        )
        pdf.savefig()
        plt.close(fig)

        fig = plt.figure()
        plt.grid()
        plt.hist(db, bins=bins)
        plt.title(
            f"{names[1]} - (N={len(db)}; M={numpy.median(db):.3f}; "
            f"$\mu$={numpy.mean(db):.3f}; $\sigma$={numpy.std(db, ddof=1):.3f})"
        )
        pdf.savefig()
        plt.close(fig)

        fig = plt.figure()
        plt.grid()
        plt.boxplot([da, db], labels=names)
        plt.title(f"Boxplots (N={len(da)})")
        pdf.savefig()
        plt.close(fig)

        fig = plt.figure()
        plt.grid()
        plt.boxplot(diff, labels=(f"{names[0]} - {names[1]}",))
        plt.title(f"Paired Differences (N={len(da)})")
        pdf.savefig()
        plt.close(fig)

        fig = plt.figure()
        plt.grid()
        plt.hist(diff, bins=bins)
        plt.title(
            f"Paired Differences "
            f"(N={len(diff)}; M={numpy.median(diff):.3f}; "
            f"$\mu$={numpy.mean(diff):.3f}; "
            f"$\sigma$={numpy.std(diff, ddof=1):.3f})"
        )
        pdf.savefig()
        plt.close(fig)

        p = scipy.stats.pearsonr(da, db)
        fig = plt.figure()
        plt.grid()
        plt.scatter(da, db, marker=".", color="black")
        plt.xlabel(f"{names[0]}")
        plt.ylabel(f"{names[1]}")
        plt.title(f"Scatter (p={p[0]:.3f})")
        pdf.savefig()
        plt.close(fig)
