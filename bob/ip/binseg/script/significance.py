#!/usr/bin/env python
# coding=utf-8

import os
import sys
import click

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
)

import numpy
import logging

logger = logging.getLogger(__name__)


from .evaluate import _validate_threshold, run as run_evaluation
from ..engine.significance import (
    patch_performances,
    visual_performances,
    write_analysis_text,
    write_analysis_figures,
    index_of_outliers,
)


def _eval_patches(
    system_name,
    threshold,
    evaluate,
    preddir,
    dataset,
    steps,
    size,
    stride,
    outdir,
    figure,
    nproc,
):
    """Calculates the patch performances on a dataset


    Parameters
    ==========

    system_name : str
        The name of the current system being analyzed

    threshold : :py:class:`float`, :py:class:`str`
        This number is used to define positives and negatives from probability
        maps, and report F1-scores (a priori). By default, we expect a set
        named 'validation' to be available at the input data.  If that is not
        the case, we use 'train', if available.  You may provide the name of
        another dataset to be used for threshold tunning otherwise.  If not
        set, or a string is input, threshold tunning is done per system,
        individually.  Optionally, you may also provide a floating-point number
        between [0.0, 1.0] as the threshold to use for both systems.

    evaluate : str
        Name of the dataset key to use from ``dataset`` to evaluate (typically,
        ``test``)

    preddir : str
        Root path to the predictions generated by system ``system_name``.  The
        final subpath inside ``preddir`` that will be used will have the value
        of this variable suffixed with the value of ``evaluate``.  We will
        search for ``<preddir>/<evaluate>/<stems>.hdf5``.

    dataset : dict
        A dictionary mapping string keys to
        :py:class:`torch.utils.data.dataset.Dataset` instances

    steps : int
        The number of threshold steps to consider when evaluating the highest
        possible F1-score on train/test data.

    size : tuple
        Two values indicating the size of windows to be used for patch
        analysis.  The values represent height and width respectively

    stride : tuple
        Two values indicating the stride of windows to be used for patch
        analysis.  The values represent height and width respectively

    outdir : str
        Path where to store visualizations.  If set to ``None``, then do not
        store performance visualizations.

    figure : str
        The name of a performance figure (e.g. ``f1_score``, or ``jaccard``) to
        use when comparing performances

    nproc : int
        Sets the number of parallel processes to use when running using
        multiprocessing.  A value of zero uses all reported cores.  A value of
        ``1`` avoids completely the use of multiprocessing and runs all chores
        in the current processing context.


    Returns
    =======

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

    if not isinstance(threshold, float):

        assert threshold in dataset, f"No dataset named '{threshold}'"

        logger.info(
            f"Evaluating threshold on '{threshold}' set for "
            f"'{system_name}' using {steps} steps"
        )
        threshold = run_evaluation(
            dataset[threshold], threshold, predictions[0], steps=steps
        )
        logger.info(f"Set --threshold={threshold:.5f} for '{system_name}'")

    # for a given threshold on each system, calculate patch performances
    logger.info(
        f"Evaluating patch performances on '{evaluate}' set for "
        f"'{system_name}' using windows of size {size} and stride {stride}"
    )

    return patch_performances(
        dataset,
        evaluate,
        preddir,
        threshold,
        size,
        stride,
        figure,
        nproc,
        outdir,
    )


def _eval_differences(perf1, perf2, evaluate, dataset, size, stride, outdir,
        figure, nproc):
    """Evaluate differences in the performance patches between two systems

    Parameters
    ----------

    perf1, perf2 : dict
        A dictionary as returned by :py:func:`_eval_patches`

    evaluate : str
        Name of the dataset key to use from ``dataset`` to evaluate (typically,
        ``test``)

    dataset : dict
        A dictionary mapping string keys to
        :py:class:`torch.utils.data.dataset.Dataset` instances

    size : tuple
        Two values indicating the size of windows to be used for patch
        analysis.  The values represent height and width respectively

    stride : tuple
        Two values indicating the stride of windows to be used for patch
        analysis.  The values represent height and width respectively

    outdir : str
        If set to ``None``, then do not output performance visualizations.
        Otherwise, in directory ``outdir``, dumps the visualizations for the
        performance differences between both systems.

    figure : str
        The name of a performance figure (e.g. ``f1_score``, or ``jaccard``) to
        use when comparing performances

    nproc : int
        Sets the number of parallel processes to use when running using
        multiprocessing.  A value of zero uses all reported cores.  A value of
        ``1`` avoids completely the use of multiprocessing and runs all chores
        in the current processing context.


    Returns
    -------

    d : dict
        A dictionary representing patch performance differences across all
        files and patches.  The format of this is similar to the individual
        inputs ``perf1`` and ``perf2``.

    """

    perf_diff = dict([(k, perf1[k]["df"].copy()) for k in perf1])

    # we can subtract these
    to_subtract = (
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "jaccard",
        "f1_score",
    )

    for k in perf_diff:
        for col in to_subtract:
            perf_diff[k][col] -= perf2[k]["df"][col]

    return visual_performances(
        dataset,
        evaluate,
        perf_diff,
        size,
        stride,
        figure,
        nproc,
        outdir,
    )


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Runs a significance test using as base the calculated predictions of two
       different systems, on the **same** dataset:
\b
       $ bob binseg significance -vv drive --names system1 system2 --predictions=path/to/predictions/system-1 path/to/predictions/system-2
\b
    2. By default, we use a "validation" dataset if it is available, to infer
       the a priori threshold for the comparison of two systems.  Otherwise,
       you may need to specify the name of a set to be used as validation set
       for choosing a threshold.  The same goes for the set to be used for
       testing the hypothesis - by default we use the "test" dataset if it is
       available, otherwise, specify.
\b
       $ bob binseg significance -vv drive --names system1 system2 --predictions=path/to/predictions/system-1 path/to/predictions/system-2 --threshold=train --evaluate=alternate-test
""",
)
@click.option(
    "--names",
    "-n",
    help="Names of the two systems to compare",
    nargs=2,
    required=True,
    type=str,
    cls=ResourceOption,
)
@click.option(
    "--predictions",
    "-p",
    help="Path where predictions of system 2 are currently stored.  You may "
    "also input predictions from a second-annotator.  This application "
    "will adequately handle it.",
    nargs=2,
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="A dictionary mapping string keys to "
    "torch.utils.data.dataset.Dataset instances",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--threshold",
    "-t",
    help="This number is used to define positives and negatives from "
    "probability maps, and report F1-scores (a priori). By default, we "
    "expect a set named 'validation' to be available at the input data. "
    "If that is not the case, we use 'train', if available.  You may provide "
    "the name of another dataset to be used for threshold tunning otherwise. "
    "If not set, or a string is input, threshold tunning is done per system, "
    "individually.  Optionally, you may also provide a floating-point number "
    "between [0.0, 1.0] as the threshold to use for both systems.",
    default="validation",
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--evaluate",
    "-e",
    help="Name of the dataset to evaluate",
    default="test",
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--steps",
    "-S",
    help="This number is used to define the number of threshold steps to "
    "consider when evaluating the highest possible F1-score on train/test data.",
    default=1000,
    type=int,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--size",
    "-s",
    help="This is a tuple with two values indicating the size of windows to "
    "be used for patch analysis.  The values represent height and width "
    "respectively.",
    default=(128, 128),
    nargs=2,
    type=int,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--stride",
    "-t",
    help="This is a tuple with two values indicating the stride of windows to "
    "be used for patch analysis.  The values represent height and width "
    "respectively.",
    default=(32, 32),
    nargs=2,
    type=int,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--figure",
    "-f",
    help="The name of a performance figure (e.g. f1_score, or jaccard) to "
    "use when comparing performances",
    default="f1_score",
    type=str,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store visualizations",
    required=False,
    type=click.Path(),
    show_default=True,
    cls=ResourceOption,
)
@click.option(
    "--remove-outliers/--no-remove-outliers",
    "-R",
    help="If set, removes outliers from both score distributions before "
    "running statistical analysis.  Outlier removal follows a 1.5 IQR range "
    "check from the difference in figures between both systems and assumes "
    "most of the distribution is contained within that range (like in a "
    "normal distribution)",
    default=False,
    required=True,
    show_default=True,
    cls=ResourceOption,
)
@click.option(
    "--remove-zeros/--no-remove-zeros",
    "-R",
    help="If set, removes instances from the statistical analysis in which "
    "both systems had a performance equal to zero.",
    default=False,
    required=True,
    show_default=True,
    cls=ResourceOption,
)
@click.option(
    "--parallel",
    "-x",
    help="Set the number of parallel processes to use when running using "
    "multiprocessing.  A value of zero uses all reported cores.",
    default=1,
    type=int,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def significance(
    names,
    predictions,
    dataset,
    threshold,
    evaluate,
    steps,
    size,
    stride,
    figure,
    output_folder,
    remove_outliers,
    remove_zeros,
    parallel,
    **kwargs,
):
    """Evaluates how significantly different are two models on the same dataset

    This application calculates the significance of results of two models
    operating on the same dataset, and subject to a priori threshold tunning.
    """

    # minimal validation to startup
    threshold = _validate_threshold(threshold, dataset)
    assert evaluate in dataset, f"No dataset named '{evaluate}'"

    perf1 = _eval_patches(
        names[0],
        threshold,
        evaluate,
        predictions[0],
        dataset,
        steps,
        size,
        stride,
        (output_folder
        if output_folder is None
        else os.path.join(output_folder, names[0])),
        figure,
        parallel,
    )

    perf2 = _eval_patches(
        names[1],
        threshold,
        evaluate,
        predictions[1],
        dataset,
        steps,
        size,
        stride,
        (output_folder
            if output_folder is None
            else os.path.join(output_folder, names[1])),
        figure,
        parallel,
    )

    perf_diff = _eval_differences(
            perf1,
            perf2,
            evaluate,
            dataset,
            size,
            stride,
            (output_folder
                if output_folder is None
                else os.path.join(output_folder, "diff")),
            figure,
            parallel,
            )

    # loads all figures for the given threshold
    stems = list(perf1.keys())
    da = numpy.array([perf1[k]["df"][figure] for k in stems]).flatten()
    db = numpy.array([perf2[k]["df"][figure] for k in stems]).flatten()
    diff = da - db

    while remove_outliers:
        outliers_diff = index_of_outliers(diff)
        if sum(outliers_diff) == 0:
            break
        diff = diff[~outliers_diff]
        da = da[~outliers_diff]
        db = db[~outliers_diff]

    if remove_zeros:
        remove_zeros = (da == 0) & (db == 0)
        diff = diff[~remove_zeros]
        da = da[~remove_zeros]
        db = db[~remove_zeros]

    if output_folder is not None:
        fname = os.path.join(output_folder, "analysis.pdf")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        write_analysis_figures(names, da, db, fname)

    if output_folder is not None:
        fname = os.path.join(output_folder, "analysis.txt")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "wt") as f:
            write_analysis_text(names, da, db, f)
    else:
        write_analysis_text(names, da, db, sys.stdout)
