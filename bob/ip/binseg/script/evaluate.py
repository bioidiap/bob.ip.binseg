#!/usr/bin/env python
# coding=utf-8

import os
import click

from bob.extension.scripts.click_helper import (
    verbosity_option,
    ConfigCommand,
    ResourceOption,
)

from ..engine.evaluator import run, compare_annotators

import logging

logger = logging.getLogger(__name__)


def _validate_threshold(t, dataset):
    """Validates the user threshold selection.  Returns parsed threshold."""

    if t is None:
        return 0.5

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


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Runs evaluation on an existing dataset configuration:
\b
       $ bob binseg evaluate -vv drive --predictions-folder=path/to/predictions --output-folder=path/to/results
\b
    2. To run evaluation on a folder with your own images and annotations, you
       must first specify resizing, cropping, etc, so that the image can be
       correctly input to the model.  Failing to do so will likely result in
       poor performance.  To figure out such specifications, you must consult
       the dataset configuration used for **training** the provided model.
       Once you figured this out, do the following:
\b
       $ bob binseg config copy csv-dataset-example mydataset.py
       # modify "mydataset.py" to your liking
       $ bob binseg evaluate -vv mydataset.py --predictions-folder=path/to/predictions --output-folder=path/to/results
""",
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store the analysis result (created if does not exist)",
    required=True,
    default="results",
    type=click.Path(),
    cls=ResourceOption,
)
@click.option(
    "--predictions-folder",
    "-p",
    help="Path where predictions are currently stored",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="A torch.utils.data.dataset.Dataset instance implementing a dataset "
    "to be used for evaluation purposes, possibly including all pre-processing "
    "pipelines required or, optionally, a dictionary mapping string keys to "
    "torch.utils.data.dataset.Dataset instances.  All keys that do not start "
    "with an underscore (_) will be processed.",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--second-annotator",
    "-S",
    help="A dataset or dictionary, like in --dataset, with the same "
    "sample keys, but with annotations from a different annotator that is "
    "going to be compared to the one in --dataset.  The same rules regarding "
    "dataset naming conventions apply",
    required=False,
    default=None,
    cls=ResourceOption,
    show_default=True,
)
@click.option(
    "--overlayed",
    "-O",
    help="Creates overlayed representations of the output probability maps, "
    "similar to --overlayed in prediction-mode, except it includes "
    "distinctive colours for true and false positives and false negatives.  "
    "If not set, or empty then do **NOT** output overlayed images.  "
    "Otherwise, the parameter represents the name of a folder where to "
    "store those",
    show_default=True,
    default=None,
    required=False,
    cls=ResourceOption,
)
@click.option(
    "--threshold",
    "-t",
    help="This number is used to define positives and negatives from "
    "probability maps, and report F1-scores (a priori). It "
    "should either come from the training set or a separate validation set "
    "to avoid biasing the analysis.  Optionally, if you provide a multi-set "
    "dataset as input, this may also be the name of an existing set from "
    "which the threshold will be estimated (highest F1-score) and then "
    "applied to the subsequent sets.  This number is also used to print "
    "the test set F1-score a priori performance",
    default=None,
    show_default=False,
    required=False,
    cls=ResourceOption,
)
@click.option(
    "--steps",
    "-S",
    help="This number is used to define the number of threshold steps to "
    "consider when evaluating the highest possible F1-score on test data.",
    default=1000,
    show_default=True,
    required=True,
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def evaluate(
    output_folder,
    predictions_folder,
    dataset,
    second_annotator,
    overlayed,
    threshold,
    steps,
    **kwargs,
):
    """Evaluates an FCN on a binary segmentation task.
    """

    threshold = _validate_threshold(threshold, dataset)

    if not isinstance(dataset, dict):
        dataset = {"test": dataset}

    if second_annotator is None:
        second_annotator = {}
    elif not isinstance(second_annotator, dict):
        second_annotator = {"test": second_annotator}
    # else, second_annotator must be a dict

    if isinstance(threshold, str):
        # first run evaluation for reference dataset, do not save overlays
        logger.info(f"Evaluating threshold on '{threshold}' set")
        threshold = run(
            dataset[threshold], threshold, predictions_folder, steps=steps
        )
        logger.info(f"Set --threshold={threshold:.5f}")

    # clean-up the overlayed path
    if overlayed is not None:
        overlayed = overlayed.strip()

    # now run with the
    for k, v in dataset.items():
        if k.startswith("_"):
            logger.info(f"Skipping dataset '{k}' (not to be evaluated)")
            continue
        logger.info(f"Analyzing '{k}' set...")
        run(
            v,
            k,
            predictions_folder,
            output_folder,
            overlayed,
            threshold,
            steps=steps,
        )
        second = second_annotator.get(k)
        if second is not None:
            if not second.all_keys_match(v):
                logger.warning(
                    f"Key mismatch between `dataset[{k}]` and "
                    f"`second_annotator[{k}]` - skipping "
                    f"second-annotator comparisons for {k} subset"
                )
            else:
                compare_annotators(v, second, k, output_folder, overlayed)
