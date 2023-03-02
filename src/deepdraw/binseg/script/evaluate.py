# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import click

from clapper.click import ConfigCommand, ResourceOption, verbosity_option
from clapper.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")


@click.command(
    entry_point_group="binseg.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
  1. Runs evaluation on an existing dataset configuration:

     .. code:: sh

        $ binseg evaluate -vv drive --predictions-folder=path/to/predictions --output-folder=path/to/results

\b
  2. To run evaluation on a folder with your own images and annotations, you
     must first specify resizing, cropping, etc, so that the image can be
     correctly input to the model.  Failing to do so will likely result in
     poor performance.  To figure out such specifications, you must consult
     the dataset configuration used for **training** the provided model.
     Once you figured this out, do the following:

     .. code:: sh

        $ binseg config copy csv-dataset-example mydataset.py
        # modify "mydataset.py" to your liking
        $ binseg evaluate -vv mydataset.py --predictions-folder=path/to/predictions --output-folder=path/to/results
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
@click.option(
    "--parallel",
    "-P",
    help="""Use multiprocessing for data processing: if set to -1 (default),
    disables multiprocessing.  Set to 0 to enable as many data loading
    instances as processing cores as available in the system.  Set to >= 1 to
    enable that many multiprocessing instances for data processing.""",
    type=click.IntRange(min=-1),
    show_default=True,
    required=True,
    default=-1,
    cls=ResourceOption,
)
@verbosity_option(logger=logger, cls=ResourceOption)
@click.pass_context
def evaluate(
    ctx,
    output_folder,
    predictions_folder,
    dataset,
    second_annotator,
    overlayed,
    threshold,
    steps,
    parallel,
    verbose,
    **kwargs,
):
    """Evaluate an FCN on a binary segmentation task."""
    from ...common.script.evaluate import base_evaluate

    ctx.invoke(
        base_evaluate,
        output_folder=output_folder,
        predictions_folder=predictions_folder,
        dataset=dataset,
        second_annotator=second_annotator,
        overlayed=overlayed,
        threshold=threshold,
        steps=steps,
        parallel=parallel,
        detection=False,
        verbose=verbose,
    )
