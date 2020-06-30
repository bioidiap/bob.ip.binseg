#!/usr/bin/env python
# coding=utf-8

import os
import click

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
from ..engine.significance import patch_performances


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Runs a significance test using as base the calculated predictions of two
       different systems (1, the first and 2, the second), on the **same**
       dataset:
\b
       $ bob binseg significance -vv drive --predictions-1=path/to/predictions/system-1 --predictions-2=path/to/predictions/system-2
\b
    2. By default, we use a "validation" dataset if it is available, to infer
       the a priori threshold for the comparison of two systems.  Otherwise,
       you may need to specify the name of a set to be used as validation set
       for choosing a threshold.  The same goes for the set to be used for
       testing the hypothesis - by default we use the "test" dataset if it is
       available, otherwise, specify.
\b
       $ bob binseg significance -vv drive --predictions-1=path/to/predictions/system-1 --predictions-2=path/to/predictions/system-2 --threshold=train --evaluate=alternate-test
""",
)
@click.option(
    "--predictions-1",
    "-p",
    help="Path where predictions of system 1 are currently stored.  You may "
         "also input predictions from a second-annotator.  This application "
         "will adequately handle it.",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    cls=ResourceOption,
)
@click.option(
    "--predictions-2",
    "-P",
    help="Path where predictions of system 2 are currently stored.  You may "
         "also input predictions from a second-annotator.  This application "
         "will adequately handle it.",
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
    default='validation',
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--evaluate",
    "-e",
    help="Name of the dataset to evaluate",
    default='test',
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
@verbosity_option(cls=ResourceOption)
def significance(
    predictions_1,
    predictions_2,
    dataset,
    threshold,
    evaluate,
    steps,
    size,
    stride,
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

    else:  #it is a string, re-calculate it for each system individually

        assert threshold in dataset, f"No dataset named '{threshold}'"

        logger.info(f"Evaluating threshold on '{threshold}' set for system 1 using {steps} steps")
        threshold1 = run_evaluation(
            dataset[threshold], threshold, predictions_1, steps=steps
        )
        logger.info(f"Set --threshold={threshold1:.5f} for system 1")

        logger.info(f"Evaluating threshold on '{threshold}' set for system 2 using {steps} steps")
        threshold2 = run_evaluation(
            dataset[threshold], threshold, predictions_2, steps=steps
        )
        logger.info(f"Set --threshold={threshold2:.5f} for system 2")

    # for a given threshold on each system, calculate patch performances
    logger.info(f"Evaluating patch performances on '{evaluate}' set for system 1 using windows of size {size} and stride {stride}")
    perf1 = patch_performances(dataset, evaluate, predictions_1, threshold1,
            size, stride)
    logger.info(f"Evaluating patch performances on '{evaluate}' set for system 2 using windows of size {size} and stride {stride}")
    perf2 = patch_performances(dataset, evaluate, predictions_2, threshold2,
            size, stride)

    ###### MAGIC STARTS #######

    # load all F1-scores for the given threshold
    da = perf1.f1_score
    db = perf2.f1_score
    diff = da - db

    import matplotlib
    import matplotlib.pyplot as plt
    plt.subplot(2,2,1)
    plt.boxplot([da, db])
    plt.title('Systems 1 and 2')
    plt.subplot(2,2,2)
    plt.boxplot(diff)
    plt.title('Differences (1 - 2)')
    plt.subplot(2,1,2)
    plt.hist(diff, bins=50)
    plt.title('Histogram (1 - 2)')
    plt.savefig('analysis.pdf')

    #diff = diff[diff!=0.0]
    #click.echo(diff)

    click.echo("#Samples/Median/Avg/Std.Dev./Normality Conf. F1-scores:")
    click.echo(f"* system1: {len(da)} / {numpy.median(da):.3f} / {numpy.mean(da):.3f} / {numpy.std(da, ddof=1):.3f} / {scipy.stats.normaltest(da)[1]}" )
    click.echo(f"* system2: {len(db)} / {numpy.median(db):.3f} / {numpy.mean(db):.3f} / {numpy.std(db, ddof=1):.3f} / {scipy.stats.normaltest(db)[1]}" )
    click.echo(f"* system1-system2: {len(diff)} / {numpy.median(diff):.3f} / {numpy.mean(diff):.3f} / {numpy.std(diff, ddof=1):.3f} / {scipy.stats.normaltest(diff)[1]}" )

    w, p = scipy.stats.ttest_rel(da, db)
    click.echo(f"Paired T-test (is the difference zero?): S = {w:g}, p = {p:.5f}")

    w, p = scipy.stats.ttest_ind(da, db, equal_var=False)
    click.echo(f"Ind. T-test (is the difference zero?): S = {w:g}, p = {p:.5f}")

    w, p = scipy.stats.wilcoxon(diff)
    click.echo(f"Wilcoxon test (is the difference zero?): W = {w:g}, p = {p:.5f}")

    w, p = scipy.stats.wilcoxon(diff, alternative="greater")
    click.echo(f"Wilcoxon test (md(system1) < md(system2)?): W = {w:g}, p = {p:.5f}")

    w, p = scipy.stats.wilcoxon(diff, alternative="less")
    click.echo(f"Wilcoxon test (md(system1) > md(system2)?): W = {w:g}, p = {p:.5f}")
