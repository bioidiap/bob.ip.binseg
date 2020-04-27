#!/usr/bin/env python
# coding=utf-8

import os
import click

from bob.extension.scripts.click_helper import (
    verbosity_option,
    AliasedGroup,
)

import pandas
import tabulate

from ..utils.plot import precision_recall_f1iso
from ..utils.table import performance_table

import logging
logger = logging.getLogger(__name__)


def _validate_threshold(t, dataset):
    """Validates the user threshold selection.  Returns parsed threshold."""

    if t is None:
        return t

    try:
        # we try to convert it to float first
        t = float(t)
        if t < 0.0 or t > 1.0:
            raise ValueError("Float thresholds must be within range [0.0, 1.0]")
    except ValueError:
        # it is a bit of text - assert dataset with name is available
        if not isinstance(dataset, dict):
            raise ValueError(
                "Threshold should be a floating-point number "
                "if your provide only a single dataset for evaluation"
            )
        if t not in dataset:
            raise ValueError(
                f"Text thresholds should match dataset names, "
                f"but {t} is not available among the datasets provided ("
                f"({', '.join(dataset.keys())})"
            )

    return t


def _load(data, threshold=None):
    """Plots comparison chart of all evaluated models

    Parameters
    ----------

    data : dict
        A dict in which keys are the names of the systems and the values are
        paths to ``metrics.csv`` style files.

    threshold : :py:class:`float`, :py:class:`str`, Optional
        A value indicating which threshold to choose for selecting a "F1-score"
        If set to ``None``, then use the maximum F1-score on that metrics file.
        If set to a floating-point value, then use the F1-score that is
        obtained on that particular threshold.  If set to a string, it should
        match one of the keys in ``data``.  It then first calculate the
        threshold reaching the maximum F1-score on that particular dataset and
        then applies that threshold to all other sets.


    Returns
    -------

    data : dict
        A dict in which keys are the names of the systems and the values are
        dictionaries that contain two keys:

        * ``df``: A :py:class:`pandas.DataFrame` with the metrics data loaded
          to
        * ``threshold``: A threshold to be used for summarization, depending on
          the ``threshold`` parameter set on the input

    """

    if isinstance(threshold, str):
        logger.info(f"Calculating threshold from maximum F1-score at "
                f"'{threshold}' dataset...")
        metrics_path = data[threshold]
        df = pandas.read_csv(metrics_path)

        maxf1 = df.f1_score.max()
        use_threshold = df.threshold[df.f1_score.idxmax()]
        logger.info(f"Dataset '*': threshold = {use_threshold:.3f}'")

    elif isinstance(threshold, float):
        use_threshold = threshold
        logger.info(f"Dataset '*': threshold = {use_threshold:.3f}'")

    names = []
    dfs = []
    thresholds = []

    # loads all data
    retval = {}
    for name, metrics_path in data.items():

        logger.info(f"Loading metrics from {metrics_path}...")
        df = pandas.read_csv(metrics_path)

        if threshold is None:
            use_threshold = df.threshold[df.f1_score.idxmax()]
            logger.info(f"Dataset '{name}': threshold = {use_threshold:.3f}'")

        retval[name] = dict(df=df, threshold=use_threshold)

    return retval


@click.command(
    epilog="""Examples:

\b
    1. Compares system A and B, with their own pre-computed metric files:
\b
       $ bob binseg compare -vv A path/to/A/metrics.csv B path/to/B/metrics.csv
""",
)
@click.argument(
        'label_path',
        nargs=-1,
        )
@click.option(
    "--output-figure",
    "-f",
    help="Path where write the output figure (any extension supported by "
    "matplotlib is possible).  If not provided, does not produce a figure.",
    required=False,
    default=None,
    type=click.Path(dir_okay=False, file_okay=True),
)
@click.option(
    "--table-format",
    "-T",
    help="The format to use for the comparison table",
    show_default=True,
    required=True,
    default="rst",
    type=click.Choice(tabulate.tabulate_formats),
)
@click.option(
    "--output-table",
    "-u",
    help="Path where write the output table. If not provided, does not write "
    "write a table to file, only to stdout.",
    required=False,
    default=None,
    type=click.Path(dir_okay=False, file_okay=True),
)
@click.option(
    "--threshold",
    "-t",
    help="This number is used to select which F1-score to use for "
    "representing a system performance.  If not set, we report the maximum "
    "F1-score in the set, which is equivalent to threshold selection a "
    "posteriori (biased estimator).  You can either set this value to a "
    "floating-point number in the range [0.0, 1.0], or to a string, naming "
    "one of the systems which will be used to calculate the threshold "
    "leading to the maximum F1-score and then applied to all other sets.",
    default=None,
    show_default=False,
    required=False,
)
@verbosity_option()
def compare(label_path, output_figure, table_format, output_table, threshold,
        **kwargs):
    """Compares multiple systems together"""

    # hack to get a dictionary from arguments passed to input
    if len(label_path) % 2 != 0:
        raise click.ClickException("Input label-paths should be doubles"
                " composed of name-path entries")
    data = dict(zip(label_path[::2], label_path[1::2]))

    threshold = _validate_threshold(threshold, data)

    # load all data metrics
    data = _load(data, threshold=threshold)

    if output_figure is not None:
        output_figure = os.path.realpath(output_figure)
        logger.info(f"Creating and saving plot at {output_figure}...")
        os.makedirs(os.path.dirname(output_figure), exist_ok=True)
        fig = precision_recall_f1iso(data, confidence=True)
        fig.savefig(output_figure)

    logger.info("Tabulating performance summary...")
    table = performance_table(data, table_format)
    click.echo(table)
    if output_table is not None:
        output_table = os.path.realpath(output_table)
        logger.info(f"Saving table at {output_table}...")
        os.makedirs(os.path.dirname(output_table), exist_ok=True)
        with open(output_table, "wt") as f:
            f.write(table)
