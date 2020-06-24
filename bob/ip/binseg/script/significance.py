#!/usr/bin/env python
# coding=utf-8

import os
import pathlib

import click
import numpy
import scipy.stats

from bob.extension.scripts.click_helper import (
    verbosity_option,
    AliasedGroup,
)

import pandas

import logging
logger = logging.getLogger(__name__)


def _get_threshold(csvfile):
    """Returns the thresholds that maximizes the F1-score from an aggreated scan

    Parameters
    ==========

    csvfile : str
        Path to a CSV file containing the aggreated performance across various
        thresholds for a validation set.


    Returns
    =======

    threshold : float
        A float threshold

    """

    df = pandas.read_csv(csvfile)
    threshold = df.threshold[df.f1_score.idxmax()]
    logger.info(f"Threshold for '{csvfile}' = {threshold:.3f}'")
    return threshold


def _find_files(path, expression):
    """Finds a files with names matching the glob expression recursively


    Parameters
    ==========

    path : str
        The base path where to search for files

    expression : str
        A glob expression to search for filenames (e.g. ``"*.csv"``)


    Returns
    =======

    l : list
        A sorted list (by name) of relative file paths found under ``path``

    """

    return sorted([k.name for k in pathlib.Path(path).rglob(expression)])


def _load_score(path, threshold):
    """Loads the specific score (at threshold "t") from an analysis file


    Parameters
    ==========

    path : string
        Path pointing to the CSV files from where to load the scores

    threshold : float
        Threshold value to use for picking the score


    Returns
    =======

    value : float
        Value representing the score at the given threshold inside the file

    """

    df = pandas.read_csv(path)
    bins = len(df)
    index = int(round(bins * threshold))
    index = min(index, len(df) - 1)  # avoids out of range indexing
    return df.f1_score[index]


@click.command(
    epilog="""Examples:

\b
    1. Measures if systems A and B are significantly different for the same
       input data, at a threshold optimized on specific validation sets.
\b
       $ bob binseg significance -vv A path/to/A/valid.csv path/to/A/test B path/to/B/valid.csv path/to/B/test
""",
)
@click.argument(
        'label_valid_path',
        nargs=-1,
        )
@verbosity_option()
def significance(label_valid_path, **kwargs):
    """Check if systems A and B are significantly different in performance

    Significance testing is done through a `Wilcoxon Test`_.
    """

    # hack to get a dictionary from arguments passed to input
    if len(label_valid_path) % 3 != 0:
        raise click.ClickException("Input label-validation-paths should be "
                " tripplets composed of name-path-path entries")

    # split input into 2 systems
    sa, sb = list(zip(*(iter(label_valid_path),)*3))

    # sanity check file lists
    fa = _find_files(sa[2], "*.csv")
    fb = _find_files(sb[2], "*.csv")
    assert fa == fb, f"List of files mismatched between '{sa[2]}' " \
        f"and '{sb[2]}' - please check"

    # now run analysis
    ta = _get_threshold(sa[1])
    tb = _get_threshold(sb[1])

    # load all F1-scores for the given threshold
    da = [_load_score(os.path.join(sa[2], k), ta) for k in fa]
    db = [_load_score(os.path.join(sb[2], k), tb) for k in fb]

    click.echo("Median F1-scores:")
    click.echo(f"* {sa[0]}: {numpy.median(da):.3f}")
    click.echo(f"* {sb[0]}: {numpy.median(db):.3f}")

    w, p = scipy.stats.wilcoxon(da, db)
    click.echo(f"Wilcoxon test (is the difference zero?): W = {w:g}, p = {p:.5f}")

    w, p = scipy.stats.wilcoxon(da, db, alternative="greater")
    click.echo(f"Wilcoxon test (md({sa[0]}) < md({sb[0]})?): W = {w:g}, p = {p:.5f}")

    w, p = scipy.stats.wilcoxon(da, db, alternative="less")
    click.echo(f"Wilcoxon test (md({sa[0]}) > md({sb[0]})?): W = {w:g}, p = {p:.5f}")
