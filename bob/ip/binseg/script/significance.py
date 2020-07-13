#!/usr/bin/env python
# coding=utf-8

import os
import sys
import click
import typing

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
)

import numpy
import scipy.stats

import logging

logger = logging.getLogger(__name__)


from .evaluate import _validate_threshold, run as run_evaluation
from ..engine.significance import patch_performances, visual_performances


def _index_of_outliers(c):
    """Finds indexes of outlines (+/- 1.5*IQR) on a pandas dataframe column"""

    iqr = c.quantile(0.75) - c.quantile(0.25)
    limits = (c.quantile(0.25) - 1.5 * iqr, c.quantile(0.75) + 1.5 * iqr)
    return (c < limits[0]) | (c > limits[1])


def _write_analysis_text(names, da, db, f):
    """Writes a text file containing the most important statistics"""

    diff = da - db
    f.write("#Samples/Median/Avg/Std.Dev./Normality Conf. F1-scores:\n")
    f.write(
        f"* {names[0]}: {len(da)}" \
        f" / {numpy.median(da):.3f}" \
        f" / {numpy.mean(da):.3f}" \
        f" / {numpy.std(da, ddof=1):.3f}\n"
    )
    f.write(
        f"* {names[1]}: {len(db)}" \
        f" / {numpy.median(db):.3f}" \
        f" / {numpy.mean(db):.3f}" \
        f" / {numpy.std(db, ddof=1):.3f}\n"
    )
    f.write(
        f"* {names[0]}-{names[1]}: {len(diff)}" \
        f" / {numpy.median(diff):.3f}" \
        f" / {numpy.mean(diff):.3f}" \
        f" / {numpy.std(diff, ddof=1):.3f}" \
        f" / gaussian? p={scipy.stats.normaltest(diff)[1]:.3f}\n"
    )

    w, p = scipy.stats.ttest_rel(da, db)
    f.write(
        f"Paired T-test (is the difference zero?): S = {w:g}, p = {p:.5f}\n"
    )

    w, p = scipy.stats.ttest_ind(da, db, equal_var=False)
    f.write(f"Ind. T-test (is the difference zero?): S = {w:g}, p = {p:.5f}\n")

    w, p = scipy.stats.wilcoxon(diff)
    f.write(
        f"Wilcoxon test (is the difference zero?): W = {w:g}, p = {p:.5f}\n"
    )

    w, p = scipy.stats.wilcoxon(diff, alternative="greater")
    f.write(
        f"Wilcoxon test (md({names[0]}) < md({names[1]})?): " \
        f"W = {w:g}, p = {p:.5f}\n"
    )

    w, p = scipy.stats.wilcoxon(diff, alternative="less")
    f.write(
        f"Wilcoxon test (md({names[0]}) > md({names[1]})?): " \
        f"W = {w:g}, p = {p:.5f}\n"
    )


def _write_analysis_figures(names, da, db, folder):
    """Writes a PDF containing most important plots for analysis"""

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    diff = da - db
    bins = 50

    fname = os.path.join(folder, "statistics.pdf")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with PdfPages(fname) as pdf:

        plt.figure()
        plt.grid()
        plt.hist(da, bins=bins)
        plt.title(
            f"{names[0]} - scores (N={len(da)}; M={numpy.median(da):.3f}; "
            f"$\mu$={numpy.mean(da):.3f}; $\sigma$={numpy.std(da, ddof=1):.3f})"
        )
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.grid()
        plt.hist(db, bins=bins)
        plt.title(
            f"{names[1]} - scores (N={len(db)}; M={numpy.median(db):.3f}; "
            f"$\mu$={numpy.mean(db):.3f}; $\sigma$={numpy.std(db, ddof=1):.3f})"
        )
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.boxplot([da, db])
        plt.title(f"{names[0]} and {names[1]} (N={len(da)})")
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.boxplot(diff)
        plt.title(f"Differences ({names[0]} - {names[1]}) (N={len(da)})")
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.grid()
        plt.hist(diff, bins=bins)
        plt.title(
            f"Systems ({names[0]} - {names[1]}) " \
            f"(N={len(diff)}; M={numpy.median(diff):.3f}; " \
            f"$\mu$={numpy.mean(diff):.3f}; " \
            f"$\sigma$={numpy.std(diff, ddof=1):.3f})"
        )
        pdf.savefig()
        plt.close()

        p = scipy.stats.pearsonr(da, db)
        plt.figure()
        plt.grid()
        plt.scatter(da, db, marker=".", color="black")
        plt.xlabel("{names[0]}")
        plt.ylabel("{names[1]}")
        plt.title(f"Scatter (p={p[0]:.3f})")
        pdf.savefig()
        plt.close()


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
    help="The name of a performance figure (e.g. f1_score) to use for "
    "for comparing performances",
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
    help="If set, removes outliers from both score distributions before " \
         "running statistical analysis",
    default=False,
    required=True,
    show_default=True,
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
    **kwargs,
):
    """Evaluates how significantly different are two models on the same dataset

    This application calculates the significance of results of two models
    operating on the same dataset, and subject to a priori threshold tunning.
    """

    # minimal validation to startup
    threshold = _validate_threshold(threshold, dataset)
    assert evaluate in dataset, f"No dataset named '{evaluate}'"

    if isinstance(threshold, float):
        threshold1 = threshold2 = threshold

    else:  # it is a string, re-calculate it for each system individually

        assert threshold in dataset, f"No dataset named '{threshold}'"

        logger.info(
            f"Evaluating threshold on '{threshold}' set for '{names[0]}' using {steps} steps"
        )
        threshold1 = run_evaluation(
            dataset[threshold], threshold, predictions[0], steps=steps
        )
        logger.info(f"Set --threshold={threshold1:.5f} for '{names[0]}'")

        logger.info(
            f"Evaluating threshold on '{threshold}' set for '{names[1]}' using {steps} steps"
        )
        threshold2 = run_evaluation(
            dataset[threshold], threshold, predictions[1], steps=steps
        )
        logger.info(f"Set --threshold={threshold2:.5f} for '{names[1]}'")

    # for a given threshold on each system, calculate patch performances
    logger.info(
        f"Evaluating patch performances on '{evaluate}' set for '{names[0]}' using windows of size {size} and stride {stride}"
    )
    dir1 = (
        os.path.join(output_folder, names[0])
        if output_folder is not None
        else None
    )
    perf1 = patch_performances(
        dataset,
        evaluate,
        predictions[0],
        threshold1,
        size,
        stride,
        figure,
        nproc=0,
        outdir=dir1,
    )

    logger.info(
        f"Evaluating patch performances on '{evaluate}' set for '{names[1]}' using windows of size {size} and stride {stride}"
    )
    dir2 = (
        os.path.join(output_folder, names[1])
        if output_folder is not None
        else None
    )
    perf2 = patch_performances(
        dataset,
        evaluate,
        predictions[1],
        threshold2,
        size,
        stride,
        figure,
        nproc=0,
        outdir=dir2,
    )

    perf_diff = dict([(k, perf1[k]["df"].copy()) for k in perf1])
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
    dirdiff = (
        os.path.join(output_folder, "diff")
        if output_folder is not None
        else None
    )
    perf_diff = visual_performances(
        dataset,
        evaluate,
        perf_diff,
        size,
        stride,
        figure,
        nproc=0,
        outdir=dirdiff,
    )

    # loads all F1-scores for the given threshold
    stems = list(perf1.keys())
    da = numpy.array([perf1[k]["df"].f1_score for k in stems]).flatten()
    db = numpy.array([perf2[k]["df"].f1_score for k in stems]).flatten()
    diff = da - db

    while remove_outliers:
        outliers_diff = _index_of_outliers(diff)
        if sum(outliers_diff) == 0:
            break
        diff = diff[~outliers_diff]
        da = da[~outliers_diff]
        db = db[~outliers_diff]

    # also remove cases in which both da and db are zero
    remove_zeros = (da == 0) & (db == 0)
    diff = diff[~remove_zeros]
    da = da[~remove_zeros]
    db = db[~remove_zeros]

    if output_folder is not None:
        _write_analysis_figures(names, da, db, output_folder)

    if output_folder is not None:
        fname = os.path.join(output_folder, "analysis.txt")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "wt") as f:
            _write_analysis_text(names, da, db, f)
    else:
        _write_analysis_text(names, da, db, sys.stdout)
