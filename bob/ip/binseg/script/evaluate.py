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


@click.command(
    entry_point_group="bob.ip.binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
    1. Runs evaluation on an existing dataset configuration:
\b
       $ bob binseg evaluate -vv m2unet drive --predictions-folder=path/to/predictions --output-folder=path/to/results
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
       $ bob binseg evaluate -vv m2unet mydataset.py --predictions-folder=path/to/predictions --output-folder=path/to/results
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
    "--second-annotator-folder",
    "-O",
    help="Path where to store the analysis result for second annotator "
    "comparisons (only used if --second-annotator is also passed)",
    required=True,
    default="second-annotator",
    type=click.Path(),
    cls=ResourceOption,
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
    "--overlay-threshold",
    "-T",
    help="If you set --overlayed, then you can provide a value to be used as "
    "threshold to be applied on probability maps and decide for positives and "
    "negatives.  This binary output will be used to define true and false "
    "positives, and false negatives for the overlay analysis.  This number "
    "should either come from the training set or a separate validation set "
    "to avoid biasing the analysis",
    default=0.5,
    type=click.FloatRange(min=0.0, max=1.0),
    show_default=True,
    required=False,
    cls=ResourceOption,
)
@verbosity_option(cls=ResourceOption)
def evaluate(
    output_folder,
    predictions_folder,
    dataset,
    second_annotator,
    second_annotator_folder,
    overlayed,
    overlay_threshold,
    **kwargs,
):
    """Evaluates an FCN on a binary segmentation task.
    """

    # if we work with dictionaries of datasets, then output evaluation
    # information into sub-directories of the output_folder
    config = {}
    if not isinstance(dataset, dict):
        config["test"] = {
            "dataset": dataset,
            "output_folder": output_folder,
            "second_annotator": second_annotator,
            "second_annotator_folder": second_annotator_folder,
        }
    else:
        for k, v in dataset.items():
            if k.startswith("_"):
                logger.info(f"Skipping dataset '{k}' (not to be evaluated)")
                continue
            config[k] = {
                "dataset": v,
                "output_folder": os.path.join(output_folder, k),
                "second_annotator": second_annotator.get(k),
                "second_annotator_folder": os.path.join(
                    second_annotator_folder, k
                ),
            }

    for k, v in config.items():
        run(
            v["dataset"],
            predictions_folder,
            v["output_folder"],
            overlayed,
            overlay_threshold,
        )
        if v["second_annotator"] is not None:
            compare_annotators(
                v["dataset"],
                v["second_annotator"],
                v["second_annotator_folder"],
                os.path.join(overlayed, "second-annotator"),
            )
