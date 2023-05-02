# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os

import click

from clapper.click import ConfigCommand, ResourceOption, verbosity_option
from clapper.logging import setup

logger = setup(__name__.split(".")[0], format="%(levelname)s: %(message)s")

from .common import save_sh_command


@click.command(
    entry_point_group="deepdraw.config",
    cls=ConfigCommand,
    epilog="""Examples:

\b
  1. Re-evaluates a pre-trained M2U-Net model with DRIVE (vessel
     segmentation), on the CPU, by running inference and evaluation on results
     from its test set:

     .. code:: sh

        $ deepdraw analyze -vv m2unet drive --weight=model.path
""",
)
@click.option(
    "--output-folder",
    "-o",
    help="Path where to store experiment outputs (created if does not exist)",
    required=True,
    type=click.Path(),
    default="results",
    cls=ResourceOption,
)
@click.option(
    "--model",
    "-m",
    help="A torch.nn.Module instance implementing the network to be trained, and then evaluated",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--dataset",
    "-d",
    help="A dictionary mapping string keys to "
    "deepdraw.data.utils.SampleList2TorchDataset's.  At least one key "
    "named 'train' must be available.  This dataset will be used for training "
    "the network model.  All other datasets will be used for prediction and "
    "evaluation. Dataset descriptions include all required pre-processing, "
    "including eventual data augmentation, which may be eventually excluded "
    "for prediction and evaluation purposes",
    required=True,
    cls=ResourceOption,
)
@click.option(
    "--second-annotator",
    "-S",
    help="A dataset or dictionary, like in --dataset, with the same "
    "sample keys, but with annotations from a different annotator that is "
    "going to be compared to the one in --dataset",
    required=False,
    default=None,
    cls=ResourceOption,
    show_default=True,
)
@click.option(
    "--batch-size",
    "-b",
    help="Number of samples in every batch (this parameter affects "
    "memory requirements for the network).  If the number of samples in "
    "the batch is larger than the total number of samples available for "
    "training, this value is truncated.  If this number is smaller, then "
    "batches of the specified size are created and fed to the network "
    "until there are no more new samples to feed (epoch is finished).  "
    "If the total number of training samples is not a multiple of the "
    "batch-size, the last batch will be smaller than the first.",
    required=True,
    show_default=True,
    default=1,
    type=click.IntRange(min=1),
    cls=ResourceOption,
)
@click.option(
    "--device",
    "-d",
    help='A string indicating the device to use (e.g. "cpu" or "cuda:0")',
    show_default=True,
    required=True,
    default="cpu",
    cls=ResourceOption,
)
@click.option(
    "--overlayed/--no-overlayed",
    "-O",
    help="Creates overlayed representations of the output probability maps, "
    "similar to --overlayed in prediction-mode, except it includes "
    "distinctive colours for true and false positives and false negatives.  "
    "If not set, or empty then do **NOT** output overlayed images.",
    show_default=True,
    default=False,
    required=False,
    cls=ResourceOption,
)
@click.option(
    "--weight",
    "-w",
    help="Path or URL to pretrained model file (.pth extension)",
    required=True,
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
@click.option(
    "--plot-limits",
    "-L",
    help="""If set, this option affects the performance comparison plots.  It
    must be a 4-tuple containing the bounds of the plot for the x and y axis
    respectively (format: x_low, x_high, y_low, y_high]).  If not set, use
    normal bounds ([0, 1, 0, 1]) for the performance curve.""",
    default=[0.0, 1.0, 0.0, 1.0],
    show_default=True,
    nargs=4,
    type=float,
    cls=ResourceOption,
)
@verbosity_option(logger=logger, cls=ResourceOption)
@click.pass_context
def analyze(
    ctx,
    model,
    output_folder,
    batch_size,
    dataset,
    second_annotator,
    device,
    overlayed,
    weight,
    steps,
    parallel,
    plot_limits,
    verbose,
    **kwargs,
):
    """Runs a complete evaluation from prediction to comparison.

    This script is just a wrapper around the individual scripts for running
    prediction and evaluating FCN models.  It organises the output in a
    preset way::

        \b
        └─ <output-folder>/
            ├── predictions/  #the prediction outputs for the train/test set
            ├── overlayed/  #the overlayed outputs for the train/test set
                ├── predictions/  #predictions overlayed on the input images
                ├── analysis/  #predictions overlayed on the input images
                ├              #including analysis of false positives, negatives
                ├              #and true positives
                └── second-annotator/  #if set, store overlayed images for the
                                    #second annotator here
            └── analysis /  #the outputs of the analysis of both train/test sets
                            #includes second-annotator "mesures" as well, if
                            # configured

    N.B.: The tool is designed to prevent analysis bias and allows one to
    provide separate subsets for training and evaluation.  Instead of using
    simple datasets, datasets for full experiment running should be
    dictionaries with specific subset names:

    * ``__train__``: dataset used for training, prioritarily.  It is typically
      the dataset containing data augmentation pipelines.
    * ``train`` (optional): a copy of the ``__train__`` dataset, without data
      augmentation, that will be evaluated alongside other sets available
    * ``*``: any other name, not starting with an underscore character (``_``),
      will be considered a test set for evaluation.

    N.B.2: The threshold used for calculating the F1-score on the test set, or
    overlay analysis (false positives, negatives and true positives overprinted
    on the original image) also follows the logic above.
    """

    command_sh = os.path.join(output_folder, "command.sh")
    if not os.path.exists(command_sh):
        # only save if experiment has not saved yet something similar
        save_sh_command(command_sh)

    # Prediction
    logger.info("Started prediction")

    from .predict import predict

    predictions_folder = os.path.join(output_folder, "predictions")
    overlayed_folder = (
        os.path.join(output_folder, "overlayed", "predictions")
        if overlayed
        else None
    )

    ctx.invoke(
        predict,
        output_folder=predictions_folder,
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        weight=weight,
        overlayed=overlayed_folder,
        parallel=parallel,
        verbose=verbose,
    )
    logger.info("Ended prediction")

    # Evaluation
    logger.info("Started evaluation")

    from .evaluate import evaluate

    overlayed_folder = (
        os.path.join(output_folder, "overlayed", "analysis")
        if overlayed
        else None
    )

    # choosing the overlayed_threshold
    if "validation" in dataset:
        threshold = "validation"
    elif "train" in dataset:
        threshold = "train"
    else:
        threshold = 0.5
    logger.info(f"Setting --threshold={threshold}...")

    analysis_folder = os.path.join(output_folder, "analysis")
    ctx.invoke(
        evaluate,
        output_folder=analysis_folder,
        predictions_folder=predictions_folder,
        dataset=dataset,
        second_annotator=second_annotator,
        overlayed=overlayed_folder,
        threshold=threshold,
        steps=steps,
        parallel=parallel,
        verbose=verbose,
    )

    logger.info("Ended evaluation")

    # Comparison
    logger.info("Started comparison")

    # compare performances on the various sets
    from .compare import compare

    systems = []
    for k, v in dataset.items():
        if k.startswith("_"):
            logger.info(f"Skipping dataset '{k}' (not to be compared)")
            continue
        candidate = os.path.join(analysis_folder, f"{k}.csv")
        if not os.path.exists(candidate):
            logger.error(
                f"Skipping dataset '{k}' "
                f"(candidate CSV file `{candidate}` does not exist!)"
            )
            continue
        systems += [k, os.path.join(analysis_folder, f"{k}.csv")]
    if second_annotator is not None:
        for k, v in second_annotator.items():
            if k.startswith("_"):
                logger.info(
                    f"Skipping second-annotator '{k}' " f"(not to be compared)"
                )
                continue
            if k not in dataset:
                logger.info(
                    f"Skipping second-annotator '{k}' "
                    f"(no equivalent `dataset[{k}]`)"
                )
                continue
            if not dataset[k].all_keys_match(v):
                logger.warning(
                    f"Skipping second-annotator '{k}' "
                    f"(keys do not match `dataset[{k}]`?)"
                )
                continue
            candidate = os.path.join(
                analysis_folder, "second-annotator", f"{k}.csv"
            )
            if not os.path.exists(candidate):
                logger.error(
                    f"Skipping second-annotator '{k}' "
                    f"(candidate CSV file `{candidate}` does not exist!)"
                )
                continue
            systems += [f"{k} (2nd. annot.)", candidate]

    output_figure = os.path.join(output_folder, "comparison.pdf")
    output_table = os.path.join(output_folder, "comparison.rst")

    ctx.invoke(
        compare,
        label_path=systems,
        output_figure=output_figure,
        output_table=output_table,
        threshold=threshold,
        plot_limits=plot_limits,
        verbose=verbose,
    )

    logger.info("Ended comparison")
